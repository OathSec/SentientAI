"use strict";
(async function () {
  const state = await boot("any");
  if (!state) return;
  const name = state.name;
  $("title").textContent = name + "’s brain";
  if (state.signed_in) $("search").classList.remove("hidden");
  let timer = null, current = "now", hardwareTimer = null, last = null;

  function section(id, ...children) {
    const node = clear($(id));
    node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  }
  function empty(text) { return el("p", { class: "muted small" }, text); }
  function kv(pairs) {
    const dl = el("dl", { class: "kv" });
    for (const [k, v] of pairs) if (v !== null && v !== undefined) dl.append(el("dt", {}, k), el("dd", {}, String(v)));
    return dl;
  }
  const WORDS = [[0.2, "a little"], [0.45, "somewhat"], [0.7, "quite"], [1.01, "very"]];
  const level = (x) => WORDS.find(([limit]) => x < limit)[1];
  const tag = (t) => el("span", { class: "tag" }, t);

  // ------------------------------------------------------------------ tabs
  function show(tab) {
    current = tab;
    for (const b of document.querySelectorAll("#tabs [role=tab]")) b.setAttribute("aria-selected", b.dataset.tab === tab ? "true" : "false");
    for (const p of document.querySelectorAll(".panel")) p.classList.toggle("hidden", p.dataset.panel !== tab);
    try { history.replaceState(null, "", "#" + tab); } catch (e) { /* ignore */ }
    if (tab === "thinking" && state.admin) loadTalk();
    if (tab === "system" && state.admin && !hardwareTimer) hardwareTimer = Hardware.mount($("hardware"), 5000);
  }
  for (const b of document.querySelectorAll("#tabs [role=tab]")) b.addEventListener("click", () => show(b.dataset.tab));
  if (state.admin) { $("tab-people").classList.remove("hidden"); $("tab-system").classList.remove("hidden"); $("talk-thinking").classList.remove("hidden"); }

  // ---------------------------------------------------------------- render
  function render(b) {
    last = b;
    $("live-state").textContent = "updated " + new Date(b.now).toLocaleTimeString();
    clear($("banner"));
    if (b.paused) $("banner").append(el("p", { class: "notice error" }, name + " is paused (kill switch engaged)."));
    $("title").textContent = name + "’s brain · awake " + duration(b.age_s) + ", started " + b.boot_count + (b.boot_count === 1 ? " time" : " times");

    const latest = b.activities[0];
    section("now",
      el("p", { class: "nomargin" }, b.body && b.body.action ? name + " is " + b.body.action + " in its world." : ""),
      b.body && b.body.thought ? el("blockquote", { class: "pre" }, b.body.thought) : null,
      kv([
        ["Last activity", latest ? latest.kind + " · " + ago(latest.at) : "none yet"],
        ["Next activity in", b.next_activity_in_s === undefined ? null : duration(b.next_activity_in_s)],
        ["World", b.world ? Math.round(b.world.size) + " m across (up to " + Math.round(b.world.max_size) + "), " + b.world.objects + " things built" : null],
      ]),
      latest && latest.intent ? el("p", { class: "small muted" }, "Then: " + latest.intent) : null);

    section("feelings",
      b.mood ? el("p", { class: "pre" }, b.mood) : empty("No overall mood yet."),
      b.feelings.length
        ? el("ul", { class: "plain" }, b.feelings.map((f) => el("li", {},
            el("div", { class: "row spread" }, el("strong", {}, level(f.intensity) + " " + f.emotion), f.private ? tag("about someone") : null),
            bar(f.valence, { min: -1, max: 1, signed: true }),
            f.cause ? el("div", { class: "small muted" }, f.cause) : null)))
        : empty("Nothing strong right now."));

    const s = b.signals;
    section("signals", kv([
      ["Since anyone talked to it", s.hours_since_contact === null ? "no one yet" : duration(s.hours_since_contact * 3600)],
      ["Usual gap between visits", duration(s.usual_gap_hours * 3600)],
      ["Isolation", (s.isolation * 100).toFixed(0) + "%"],
      ["Sameness lately", s.sameness === null ? "not enough yet" : (s.sameness * 100).toFixed(0) + "%"],
      ["New input, last 6 h", s.stimulation],
      ["Unread things taught", s.unread],
      ["Energy", (b.affect.energy * 100).toFixed(0) + "%"],
      ["Curiosity", (b.affect.curiosity * 100).toFixed(0) + "%"],
      ["Confidence", (b.affect.competence * 100).toFixed(0) + "%"],
    ]), s.missing && s.missing.length
      ? el("p", { class: "small muted" }, "Missing: " + s.missing.map((m) => m.name + " (" + duration(m.hours * 3600) + ")").join(", "))
      : null,
      el("p", { class: "small muted" }, "Measured. How it feels about them is up to " + name + "."));

    section("goals", b.goals.length
      ? el("ul", { class: "plain" }, b.goals.map((g) => el("li", {}, g.description, " ", tag(g.origin), g.private ? tag("private") : null, bar(g.progress))))
      : empty("No goals right now."));

    section("beliefs",
      b.beliefs.length
        ? el("ul", { class: "plain" }, b.beliefs.map((x) => el("li", {},
            el("strong", {}, x.aspect + ": "), x.value,
            x.reasoning ? el("div", { class: "small muted pre" }, "because " + x.reasoning) : null,
            el("div", { class: "small muted" }, "since " + clock(x.since) + (x.source ? " · decided in: " + x.source : "")))))
        : empty(name + " hasn’t decided anything about itself yet."),
      Array.isArray(b.considering) && b.considering.length
        ? el("p", { class: "small" }, "Thinking over: " + b.considering.map((c) => c.aspect + " → " + c.value).join("; "))
        : typeof b.considering === "number" && b.considering ? el("p", { class: "small muted" }, "Thinking over " + b.considering + " possible change(s).") : null);
    section("story", b.story ? el("p", { class: "pre" }, b.story) : empty("Nothing written yet."));

    // Runs of the same choice are shown once, with how many times and over how long.
    const runs = [];
    for (const a of b.activities) {
      const last = runs[runs.length - 1];
      const key = a.kind + ":" + a.intent.trim().toLowerCase();
      if (last && last.key === key) last.items.push(a); else runs.push({ key, items: [a] });
    }
    section("activities", runs.length
      ? el("ul", { class: "plain" }, runs.map(({ items }) => {
          const a = items[0], n = items.length;
          return el("li", {},
            el("div", {}, tag(a.kind), n > 1 ? tag("×" + n) : null, a.status !== "done" ? tag(a.status) : null, a.private ? tag("private") : null, a.intent),
            a.reason ? el("div", { class: "small muted pre" }, "why: " + a.reason) : null,
            a.outcome ? el("details", {}, el("summary", {}, n > 1 ? "latest result" : "result"), el("div", { class: "small pre scroll" }, a.outcome)) : null,
            el("div", { class: "small muted" }, n > 1 ? n + " times, " + ago(items[n - 1].at) + " to " + ago(a.at) : ago(a.at)));
        }))
      : empty("Nothing yet."));

    section("projects", b.projects && b.projects.length
      ? el("div", { class: "cols two" }, b.projects.map((p) => el("div", { class: "card" },
          el("div", { class: "row spread" }, el("strong", {}, p.title), tag(p.status)),
          el("div", { class: "small muted" }, p.sessions + (p.sessions === 1 ? " session" : " sessions") + " · started " + ago(p.started) + " · last " + ago(p.updated)),
          p.why ? el("p", { class: "small pre" }, "Why: " + p.why) : null,
          p.summary ? el("p", { class: "pre" }, p.summary) : null,
          p.next_step ? el("p", { class: "small" }, el("strong", {}, "Next: "), p.next_step) : null,
          p.entries.length ? el("details", {}, el("summary", {}, "Notebook (" + p.entries.length + ")"),
            el("ul", { class: "plain" }, p.entries.map((e) => el("li", {},
              el("div", { class: "small muted" }, clock(e.at)), el("div", { class: "pre" }, e.text))))) : null)))
      : empty(name + " hasn’t started a project of its own yet. When it decides to, it’ll show up here."));

    const reasoned = b.activities.filter((a) => a.thinking);
    section("reasoning", reasoned.length
      ? el("ul", { class: "plain" }, reasoned.map((a) => el("li", {},
          el("div", {}, tag(a.kind), a.status !== "done" ? tag(a.status) : null, a.private ? tag("private") : null, a.intent),
          el("details", {}, el("summary", {}, "how it got there"), el("pre", { class: "thought" }, a.thinking)),
          el("div", { class: "small muted" }, ago(a.at)))))
      : empty("Nothing recorded yet. This fills in as it decides what to do."));

    section("thoughts", b.thoughts.length
      ? el("ul", { class: "plain" }, b.thoughts.map((t) => el("li", {},
          tag(t.kind), t.private ? tag("private") : null,
          el("div", { class: "pre" }, t.content),
          el("div", { class: "small muted" }, ago(t.at)))))
      : empty("Nothing yet."));

    const kinds = Object.entries(b.memory.by_kind).map(([k, n]) => n + " " + k).join(", ");
    $("memory-stats").textContent = b.memory.total + " memories" + (kinds ? " (" + kinds + ")" : "");

    if (b.viewer.admin) {
      section("people", b.people && b.people.length
        ? el("ul", { class: "plain" }, b.people.map((p) => el("li", {},
            el("strong", {}, (p.user.display_name || p.user.id) + (p.nickname ? " (“" + p.nickname + "”)" : "")),
            el("div", { class: "small muted" }, p.interactions + " conversations, last " + ago(p.last_seen_at)),
            el("div", { class: "pre" }, p.opinion || "no opinion yet"),
            el("div", { class: "small muted pre" }, p.feelings || ""),
            el("div", { class: "row small" }, "affinity ", bar(p.affinity, { min: -1, max: 1, signed: true }), "trust ", bar(p.trust)))))
        : empty("Nobody yet."));
      section("work",
        kv([
          ["Tasks", Object.entries(b.tasks || {}).map(([k, v]) => v + " " + k).join(", ") || "none"],
          ["Waiting for approval", b.pending_approvals],
          ["Models", Object.entries(b.models || {}).map(([k, v]) => k + (v ? " up" : " DOWN")).join(", ")],
        ]),
        b.awareness && b.awareness.length
          ? el("ul", { class: "plain" }, b.awareness.map((a) => el("li", { class: "small" }, tag(a.source), a.content)))
          : empty("Nothing competing for attention."));
    }
    const feed = $("feed");
    if (!feed.textContent && b.events) for (const e of b.events) addEvent(e, true);
  }

  async function loadTalk() {
    try {
      const rows = await api("/api/admin/thinking?limit=40");
      const byTask = new Map();
      for (const r of rows) {
        const key = r.task_id || "?";
        if (!byTask.has(key)) byTask.set(key, { at: r.at, items: [] });
        byTask.get(key).items.push(r);
      }
      const groups = [...byTask.values()].reverse();
      section("talk", groups.length ? el("ul", { class: "plain" }, groups.map((g) => el("li", {},
        el("div", { class: "small muted" }, clock(g.at)),
        g.items.map((r) => {
          if (r.type === "reasoning") return el("details", {}, el("summary", {}, "what it thought first"), el("pre", { class: "thought" }, r.reasoning || ""));
          if (r.type === "thought") return el("div", { class: "small pre" }, "while working: " + (r.text || ""));
          if (r.type === "grounding") {
            const issues = [].concat(r.unsupported || [], r.contradictions || []);
            return issues.length ? el("div", { class: "small warn pre" }, "checked its draft and found: " + issues.join("; ") + (r.revised ? " (rewrote the reply)" : "")) : null;
          }
          return el("div", {}, el("span", { class: "muted small" }, "said: "), Markdown.render(r.answer || ""));
        })))) : empty("No conversations yet."));
    } catch (e) { section("talk", el("p", { class: "notice error" }, e.message)); }
  }

  function summarize(e) {
    const p = e.payload || {};
    switch (e.type) {
      case "mind.thought": return "thought: " + p.thought;
      case "mind.feeling": return "feels " + p.emotion + " (" + Number(p.intensity).toFixed(2) + ")" + (p.cause ? ": " + p.cause : "");
      case "mind.activity": return p.kind + " [" + p.status + "] " + p.intent;
      case "world.body": return "world: " + (p.body && p.body.action);
      case "world.object": return "world: " + p.op + " " + (p.object && p.object.name || "");
      case "world.avatar": return "world: changed how it looks";
      case "world.size": return "world: grew to " + p.size + " m";
      case "mind.project": return "project " + (p.op === "started" ? "started" : "notebook") + ": " + p.title + (p.op === "entry" && p.entry ? " - " + p.entry : "");
      case "self.belief_formed": case "self.belief_considering": case "self.belief_revised": case "self.belief_kept":
        return e.type.replace("self.belief_", "self: ") + " " + p.aspect + " = " + p.value;
      case "social.appraised": return "opinion of someone updated";
      default: return e.type + " " + JSON.stringify(p).slice(0, 400);
    }
  }
  function addEvent(e, quiet) {
    if (e.type === "world.body") return;
    const feed = $("feed");
    feed.textContent = new Date(e.ts).toLocaleTimeString() + "  " + summarize(e) + "\n" + feed.textContent;
    if (feed.textContent.length > 40000) feed.textContent = feed.textContent.slice(0, 40000);
    if (!quiet) schedule();
  }

  async function refresh() {
    try { render(await api("/api/brain")); } catch (e) { $("live-state").textContent = e.message; }
    if (current === "thinking" && state.admin) loadTalk();
  }
  function schedule() { if (timer) return; timer = setTimeout(() => { timer = null; refresh(); }, 1500); }

  $("search").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("q").value.trim();
    if (q.length < 2) return;
    const box = clear($("memory-results"));
    try {
      const hits = await api("/api/brain/memories?q=" + encodeURIComponent(q));
      box.append(hits.length ? el("ul", { class: "plain" }, hits.map((h) => el("li", {},
        tag(h.kind), h.private ? tag("private") : null,
        el("div", { class: "pre" }, h.content),
        el("div", { class: "small muted" }, clock(h.at) + " · relevance " + h.relevance)))) : empty("Nothing matching."));
    } catch (err) { box.append(el("p", { class: "notice error" }, err.message)); }
  });

  await refresh();
  const wanted = location.hash.slice(1);
  show(document.querySelector('#tabs [data-tab="' + wanted + '"]:not(.hidden)') ? wanted : "now");
  live("/api/brain/stream", (e) => addEvent(e, false));
  setInterval(refresh, 30000);
})();
