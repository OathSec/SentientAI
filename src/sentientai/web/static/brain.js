"use strict";
(async function () {
  const state = await boot("any");
  if (!state) return;
  const name = state.name;
  $("title").textContent = name + "’s brain";
  if (state.signed_in) $("search").classList.remove("hidden");
  let timer = null;

  function section(id, ...children) { const node = clear($(id)); node.append(...children); }
  function empty(text) { return el("p", { class: "muted" }, text); }
  function kv(pairs) {
    const dl = el("dl", { class: "kv" });
    for (const [k, v] of pairs) if (v !== null && v !== undefined) dl.append(el("dt", {}, k), el("dd", {}, String(v)));
    return dl;
  }
  const WORDS = [[0.2, "a little"], [0.45, "somewhat"], [0.7, "quite"], [1.01, "very"]];
  const level = (x) => WORDS.find(([limit]) => x < limit)[1];

  function render(b) {
    $("live-state").textContent = "updated " + new Date(b.now).toLocaleTimeString();
    clear($("banner"));
    if (b.paused) $("banner").append(el("p", { class: "notice error" }, name + " is paused (kill switch engaged)."));
    $("title").textContent = name + "’s brain – awake " + duration(b.age_s) + ", started " + b.boot_count + (b.boot_count === 1 ? " time" : " times");

    section("feelings",
      b.mood ? el("p", {}, b.mood) : empty("No overall mood yet."),
      b.feelings.length
        ? el("ul", { class: "plain" }, b.feelings.map((f) => el("li", {},
            el("div", { class: "row spread" }, el("strong", {}, level(f.intensity) + " " + f.emotion), f.private ? el("span", { class: "tag" }, "about someone") : null),
            bar(f.valence, { min: -1, max: 1, signed: true }),
            f.cause ? el("div", { class: "small muted" }, f.cause) : null)))
        : empty("Nothing strong right now."));

    const s = b.signals;
    section("signals", kv([
      ["Since anyone talked to it", s.hours_since_contact === null ? "no one yet" : duration(s.hours_since_contact * 3600)],
      ["Usual gap between visits", duration(s.usual_gap_hours * 3600)],
      ["Isolation", (s.isolation * 100).toFixed(0) + "%"],
      ["Sameness of recent activity", s.sameness === null ? "not enough yet" : (s.sameness * 100).toFixed(0) + "%"],
      ["New input, last 6 h", s.stimulation],
      ["Unread things taught", s.unread],
      ["Energy", (b.affect.energy * 100).toFixed(0) + "%"],
      ["Curiosity", (b.affect.curiosity * 100).toFixed(0) + "%"],
      ["Confidence", (b.affect.competence * 100).toFixed(0) + "%"],
    ]), s.missing && s.missing.length
      ? el("p", { class: "small muted" }, "Missing: " + s.missing.map((m) => m.name + " (" + duration(m.hours * 3600) + ")").join(", "))
      : null,
      el("p", { class: "small muted" }, "These are measured. How it feels about them is up to " + name + "."));

    section("beliefs",
      b.beliefs.length
        ? el("ul", { class: "plain" }, b.beliefs.map((x) => el("li", {},
            el("strong", {}, x.aspect + ": "), x.value,
            x.reasoning ? el("div", { class: "small muted" }, "because " + x.reasoning) : null,
            el("div", { class: "small muted" }, "since " + clock(x.since)))))
        : empty(name + " hasn’t decided anything about itself yet."),
      Array.isArray(b.considering) && b.considering.length
        ? el("p", { class: "small" }, "Thinking over: " + b.considering.map((c) => c.aspect + " → " + c.value).join("; "))
        : typeof b.considering === "number" && b.considering ? el("p", { class: "small muted" }, "Thinking over " + b.considering + " possible change(s).") : null,
      b.story ? el("details", {}, el("summary", {}, "Its story so far"), el("p", {}, b.story)) : null);

    section("goals", b.goals.length
      ? el("ul", { class: "plain" }, b.goals.map((g) => el("li", {}, g.description, " ",
          el("span", { class: "tag" }, g.origin), g.private ? el("span", { class: "tag" }, "private") : null,
          bar(g.progress))))
      : empty("No goals right now."));

    section("activities", b.activities.length
      ? el("ul", { class: "plain" }, b.activities.map((a) => el("li", {},
          el("div", {}, el("span", { class: "tag" }, a.kind), a.status !== "done" ? el("span", { class: "tag" }, a.status) : null, a.private ? el("span", { class: "tag" }, "private") : null, a.intent),
          a.reason ? el("div", { class: "small muted" }, "why: " + a.reason) : null,
          a.outcome ? el("details", {}, el("summary", {}, "result"), el("div", { class: "small pre" }, a.outcome)) : null,
          el("div", { class: "small muted" }, ago(a.at)))))
      : empty("Nothing yet."));

    section("thoughts", b.thoughts.length
      ? el("ul", { class: "plain" }, b.thoughts.map((t) => el("li", {},
          el("span", { class: "tag" }, t.kind), t.private ? el("span", { class: "tag" }, "private") : null,
          el("div", { class: "pre" }, t.content),
          el("div", { class: "small muted" }, ago(t.at)))))
      : empty("Nothing yet."));

    const kinds = Object.entries(b.memory.by_kind).map(([k, n]) => n + " " + k).join(", ");
    $("memory-stats").textContent = b.memory.total + " memories" + (kinds ? " (" + kinds + ")" : "");

    if (b.viewer.admin) {
      $("admin").classList.remove("hidden");
      section("people", b.people && b.people.length
        ? el("ul", { class: "plain" }, b.people.map((p) => el("li", {},
            el("strong", {}, (p.user.display_name || p.user.id) + (p.nickname ? " (“" + p.nickname + "”)" : "")),
            el("div", { class: "small muted" }, p.interactions + " conversations, last " + ago(p.last_seen_at)),
            el("div", {}, p.opinion || "no opinion yet"),
            el("div", { class: "small muted" }, p.feelings || ""),
            el("div", { class: "row small" }, "affinity ", bar(p.affinity, { min: -1, max: 1, signed: true }), "trust ", bar(p.trust)))))
        : empty("Nobody yet."));
      section("work",
        kv([
          ["Next activity in", duration(b.next_beat_in_s || 0)],
          ["Tasks", Object.entries(b.tasks || {}).map(([k, v]) => v + " " + k).join(", ") || "none"],
          ["Waiting for approval", b.pending_approvals],
          ["Models", Object.entries(b.models || {}).map(([k, v]) => k + (v ? " up" : " DOWN")).join(", ")],
        ]),
        b.awareness && b.awareness.length
          ? el("ul", { class: "plain" }, b.awareness.map((a) => el("li", { class: "small" }, el("span", { class: "tag" }, a.source), a.content)))
          : empty("Nothing competing for attention."));
    }
    const feed = $("feed");
    if (!feed.textContent && b.events) for (const e of b.events) addEvent(e, true);
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
      case "self.belief_formed": case "self.belief_considering": case "self.belief_revised": case "self.belief_kept":
        return e.type.replace("self.belief_", "self: ") + " " + p.aspect + " = " + p.value;
      case "social.appraised": return "opinion of someone updated";
      default: return e.type + " " + JSON.stringify(p).slice(0, 160);
    }
  }
  function addEvent(e, quiet) {
    if (e.type === "world.body") return;
    const feed = $("feed");
    feed.textContent = new Date(e.ts).toLocaleTimeString() + "  " + summarize(e) + "\n" + feed.textContent;
    if (feed.textContent.length > 12000) feed.textContent = feed.textContent.slice(0, 12000);
    if (!quiet) schedule();
  }

  async function refresh() { try { render(await api("/api/brain")); } catch (e) { $("live-state").textContent = e.message; } }
  function schedule() { if (timer) return; timer = setTimeout(() => { timer = null; refresh(); }, 1500); }

  $("search").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("q").value.trim();
    if (q.length < 2) return;
    const box = clear($("memory-results"));
    try {
      const hits = await api("/api/brain/memories?q=" + encodeURIComponent(q));
      box.append(hits.length ? el("ul", { class: "plain" }, hits.map((h) => el("li", {},
        el("span", { class: "tag" }, h.kind), h.private ? el("span", { class: "tag" }, "private") : null,
        el("div", { class: "pre" }, h.content),
        el("div", { class: "small muted" }, clock(h.at) + " · relevance " + h.relevance)))) : empty("Nothing matching."));
    } catch (err) { box.append(el("p", { class: "notice error" }, err.message)); }
  });

  await refresh();
  live("/api/brain/stream", (e) => addEvent(e, false));
  setInterval(refresh, 30000);
})();
