"use strict";
(async function () {
  const state = await boot("user");
  if (!state) return;
  const name = state.name;
  const me = state.user.display_name;
  const log = $("log"), status = $("status"), text = $("text"), send = $("send");
  let oldest = null, busy = false;

  function message(m) {
    const fuzzy = m.role === "assistant";
    const node = el("div", { class: "msg " + (fuzzy ? "fuzzy" : "me") },
      el("div", { class: "head" }, el("span", { class: "name" }, fuzzy ? name : me), clock(m.at)),
      el("div", { class: "text" }, m.content));
    for (const image of m.media || []) {
      node.append(el("img", { src: image.url, alt: "An image from " + name, loading: "lazy" }));
    }
    return node;
  }

  async function loadHistory(before) {
    const query = "?limit=40" + (before ? "&before=" + encodeURIComponent(before) : "");
    const data = await api("/api/chat/history" + query);
    if (data.messages.length) oldest = data.messages[0].at;
    const fragment = document.createDocumentFragment();
    for (const m of data.messages) fragment.append(message(m));
    log.insertBefore(fragment, before ? log.firstChild : null);
    $("earlier").classList.toggle("hidden", !data.more);
    if (!before) {
      if (!data.messages.length) log.append(el("p", { class: "muted", id: "empty" }, "This is the start of your conversation with " + name + "."));
      window.scrollTo(0, document.body.scrollHeight);
    }
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
      $("mood").textContent = brain.paused ? name + " is paused right now." : brain.mood ? name + " is feeling: " + brain.mood : "";
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
    if (!content || busy) return;
    busy = true; send.disabled = true;
    const empty = $("empty"); if (empty) empty.remove();
    log.append(message({ role: "user", content, at: new Date().toISOString() }));
    text.value = "";
    window.scrollTo(0, document.body.scrollHeight);
    const started = Date.now();
    const timer = setInterval(() => { status.textContent = name + " is thinking… " + Math.round((Date.now() - started) / 1000) + " s"; }, 500);
    try {
      const reply = await api("/api/chat", { method: "POST", body: { message: content } });
      log.append(message({ role: "assistant", content: reply.answer, at: reply.at, media: reply.media }));
      status.textContent = "";
    } catch (err) {
      status.textContent = "";
      log.append(el("p", { class: "notice error" }, err.message));
    } finally {
      clearInterval(timer); busy = false; send.disabled = false; text.focus();
      window.scrollTo(0, document.body.scrollHeight);
      api("/api/me").then(showLimits).catch(() => {});
    }
  });
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("form").requestSubmit(); }
  });

  await Promise.all([loadHistory(null), loadNotes(), loadMood(), api("/api/me").then(showLimits)]);
  text.focus();
})();
