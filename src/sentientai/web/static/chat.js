"use strict";
(async function () {
  const state = await boot("user");
  if (!state) return;
  const name = state.name;
  const log = $("log"), status = $("status"), text = $("text"), send = $("send");
  let cursor = null;          // time of the newest message shown (server time)
  let oldest = null;
  let pending = null;         // the optimistic bubble for a message the server hasn't echoed yet
  let typing = null;          // the "Fuzzy is typing" bubble
  let polling = null;
  let lastDay = null;

  const dayKey = (iso) => new Date(iso).toDateString();
  const hhmm = (iso) => new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const nearBottom = () => log.scrollTop + log.clientHeight >= log.scrollHeight - 160;
  const toBottom = () => { log.scrollTop = log.scrollHeight; };

  function bubble(m, extra) {
    const mine = m.role !== "assistant";
    const body = mine ? m.content : Markdown.render(m.content);
    const node = el("div", { class: "msg " + (mine ? "me" : "fuzzy") + (extra ? " " + extra : "") },
      el("div", { class: "bubble" + (mine ? "" : " rich") }, body));
    for (const image of m.media || []) {
      const link = el("a", { href: image.url, target: "_blank", rel: "noopener" },
        el("img", { src: image.url, alt: "An image from " + name, loading: "lazy" }));
      node.firstChild.append(link);
    }
    node.append(el("div", { class: "stamp" }, extra === "pending" ? "sending…" : hhmm(m.at)));
    return node;
  }
  function dayLabel(iso) {
    return el("div", { class: "day" }, new Date(iso).toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" }));
  }

  function append(messages) {
    const stick = nearBottom();
    for (const m of messages) {
      if (cursor && m.at <= cursor) continue;
      if (pending && m.role === "user" && pending.dataset.text === m.content) { pending.remove(); pending = null; }
      if (typing) typing.remove();
      if (dayKey(m.at) !== lastDay) { log.append(dayLabel(m.at)); lastDay = dayKey(m.at); }
      log.append(bubble(m));
      if (typing) log.append(typing);
      cursor = m.at;
    }
    if (stick) toBottom();
  }

  function showTyping(since) {
    if (typing) return;
    typing = el("div", { class: "msg fuzzy typing" }, el("div", { class: "bubble" }, el("span", { class: "dot" }), el("span", { class: "dot" }), el("span", { class: "dot" })));
    log.append(typing);
    if (nearBottom()) toBottom();
    const started = since ? new Date(since).getTime() : Date.now();
    typing._timer = setInterval(() => {
      const s = Math.round((Date.now() - started) / 1000);
      status.textContent = s >= 8 ? name + " is thinking… " + s + " s. You can leave; the reply will be here when you come back." : "";
    }, 1000);
  }
  function hideTyping() {
    if (typing) { clearInterval(typing._timer); typing.remove(); typing = null; }
    status.textContent = "";
  }

  function startPolling(since) {
    showTyping(since);
    if (polling) return;
    polling = setInterval(async () => {
      try {
        const r = await api("/api/chat/poll?after=" + encodeURIComponent(cursor || new Date(0).toISOString()));
        append(r.messages);
        if (!r.busy) stopPolling();
      } catch (e) { /* try again on the next tick */ }
    }, 1500);
  }
  function stopPolling() {
    clearInterval(polling); polling = null; hideTyping();
    send.disabled = false;
    api("/api/me").then(showLimits).catch(() => {});
    if (document.hasFocus()) text.focus();
  }

  async function loadHistory(before) {
    const query = "?limit=40" + (before ? "&before=" + encodeURIComponent(before) : "");
    const data = await api("/api/chat/history" + query);
    if (data.messages.length) oldest = data.messages[0].at;
    $("earlier").classList.toggle("hidden", !data.more);
    if (before) {
      const fragment = document.createDocumentFragment();
      for (const m of data.messages) fragment.append(bubble(m));
      log.insertBefore(fragment, log.firstChild);
      return data;
    }
    for (const m of data.messages) {
      if (dayKey(m.at) !== lastDay) { log.append(dayLabel(m.at)); lastDay = dayKey(m.at); }
      log.append(bubble(m));
      cursor = m.at;
    }
    if (!data.messages.length) log.append(el("p", { class: "muted day" }, "This is the start of your conversation with " + name + "."));
    toBottom();
    return data;
  }

  async function loadNotes() {
    const data = await api("/api/notes");
    const unread = data.notes.filter((n) => !n.read);
    const box = clear($("notes"));
    if (!unread.length) return;
    const card = el("div", { class: "card" }, el("h2", {}, name + " left you a note"));
    for (const n of unread) card.append(el("p", { class: "pre" }, n.content), el("p", { class: "small muted" }, clock(n.at)));
    card.append(el("button", { class: "secondary", onclick: async () => { await api("/api/notes/read", { method: "POST" }); box.removeChild(card); } }, "Got it"));
    box.append(card);
  }

  async function loadMood() {
    try {
      const brain = await api("/api/brain");
      const doing = brain.body && brain.body.action && brain.body.action !== "standing" ? brain.body.action : "";
      $("mood").textContent = brain.paused ? name + " is paused right now."
        : [brain.mood ? name + " is feeling: " + brain.mood : "", doing ? "(" + doing + " in its world)" : ""].filter(Boolean).join(" ");
    } catch (e) { /* the brain page is optional here */ }
  }

  function showLimits(info) {
    const used = info.usage.messages, limit = info.limits.messages;
    $("limits").textContent = limit && !state.admin ? used + " of " + limit + " messages today" : "";
  }

  $("earlier").addEventListener("click", () => loadHistory(oldest));

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const content = text.value.trim();
    if (!content || send.disabled) return;
    send.disabled = true;
    const empty = log.querySelector("p.day"); if (empty) empty.remove();
    pending = bubble({ role: "user", content, at: new Date().toISOString() }, "pending");
    pending.dataset.text = content;
    log.append(pending);
    text.value = "";
    toBottom();
    try {
      const r = await api("/api/chat", { method: "POST", body: { message: content } });
      startPolling(r.at);
    } catch (err) {
      pending.remove(); pending = null;
      text.value = content;
      status.textContent = err.message;
      send.disabled = false;
    }
  });
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("form").requestSubmit(); }
  });
  // Coming back to the tab: catch up on anything said while away.
  document.addEventListener("visibilitychange", async () => {
    if (document.hidden || polling || !cursor) return;
    try { const r = await api("/api/chat/poll?after=" + encodeURIComponent(cursor)); append(r.messages); if (r.busy) { send.disabled = true; startPolling(r.since); } } catch (e) { /* ignore */ }
  });

  const [history] = await Promise.all([loadHistory(null), loadNotes(), loadMood(), api("/api/me").then(showLimits)]);
  if (history.busy) {
    // Fuzzy is still working on something you sent before you left.
    send.disabled = true;
    const r = await api("/api/chat/poll?after=" + encodeURIComponent(cursor || new Date(0).toISOString()));
    append(r.messages);
    if (r.busy) startPolling(r.since); else send.disabled = false;
  }
  text.focus();
  setInterval(() => { if (!document.hidden) loadMood(); }, 30000);
})();
