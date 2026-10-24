"use strict";
/* Live numbers from Fuzzy. Aggregates only; if Fuzzy is unreachable the block just says so. */
(function () {
  const root = document.getElementById("live");
  if (!root) return;
  const API = "https://entertainment.systembridge.site/api/public/stats";
  const fmt = (n) => Number(n).toLocaleString();
  const span = (h) => (h < 1 ? Math.max(1, Math.round(h * 60)) + " min" : (h < 10 ? h.toFixed(1) : Math.round(h)) + " h");
  const DOING = {
    standing: "resting for a moment", resting: "resting", thinking: "thinking something over",
    building: "building in its world", walking: "taking a walk through its world",
    creating: "painting something", talking: "chatting with someone",
  };
  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const tile = (value, label) => { const t = node("div", "tile"); t.append(node("div", "num", value), node("div", "lab", label)); return t; };
  function render(d) {
    const f = d.fuzzy, c = d.compute;
    document.getElementById("live-doing").textContent = "Right now Fuzzy is " + (DOING[f.doing] || f.doing || "resting") + ".";
    document.getElementById("live-life").replaceChildren(
      tile(fmt(f.days_alive), f.days_alive === 1 ? "day alive" : "days alive"), tile(fmt(f.memories), "memories"),
      tile(fmt(f.things_it_did), "things it has done"), tile(fmt(f.pictures), "pictures painted"),
      tile(fmt(f.things_built), "things built"), tile(fmt(f.people), "people it knows"));
    document.getElementById("live-power").replaceChildren(
      tile(fmt(c.computers_online), c.computers_online === 1 ? "computer online" : "computers online"),
      tile(fmt(c.contributors), c.contributors === 1 ? "contributor" : "contributors"),
      tile(fmt(c.jobs), "jobs done for Fuzzy"), tile(span(c.hours), "of GPU time lent"));
    const board = document.getElementById("live-board");
    const top = Math.max(0.0001, ...d.leaders.map((l) => l.hours));
    board.replaceChildren(...(d.leaders.length ? d.leaders.map((l, i) => {
      const li = node("li", "row");
      const bar = node("span", "fill"); bar.style.width = Math.max(4, (l.hours / top) * 100) + "%";
      const track = node("span", "track"); track.append(bar);
      li.append(node("span", "rank", String(i + 1)), node("span", "who", l.name), track, node("span", "stat", span(l.hours) + " · " + fmt(l.jobs) + " jobs"));
      return li;
    }) : [node("li", "empty", "Nobody on the board yet. Be the first to lend a GPU.")]));
    root.hidden = false;
  }
  async function load() {
    const note = document.getElementById("live-note");
    try {
      const r = await fetch(API, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      render(await r.json());
      note.textContent = "";
    } catch (e) {
      root.hidden = false;
      note.textContent = "Fuzzy is resting right now, so live numbers are unavailable.";
    }
  }
  load();
  setInterval(() => { if (!document.hidden) load(); }, 15000);
})();
