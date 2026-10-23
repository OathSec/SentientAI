"use strict";
/* Live numbers from Fuzzy. Aggregates only; if Fuzzy is unreachable the block just says so. */
(function () {
  const root = document.getElementById("live");
  if (!root) return;
  const API = "https://entertainment.systembridge.site/api/public/stats";
  const fmt = (n) => Number(n).toLocaleString();
  const tile = (label, value) => {
    const d = document.createElement("div"); d.className = "tile";
    const n = document.createElement("div"); n.className = "num"; n.textContent = value;
    const l = document.createElement("div"); l.className = "small muted"; l.textContent = label;
    d.append(n, l); return d;
  };
  function render(d) {
    const f = d.fuzzy, c = d.compute;
    const tiles = document.getElementById("live-tiles"), board = document.getElementById("live-board");
    tiles.replaceChildren(
      tile("days alive", fmt(f.days_alive)), tile("memories", fmt(f.memories)), tile("things it has done", fmt(f.things_it_did)),
      tile("pictures painted", fmt(f.pictures)), tile("things built", fmt(f.things_built)), tile("people it knows", fmt(f.people)),
      tile("computers online now", fmt(c.computers_online)), tile("contributors", fmt(c.contributors)),
      tile("jobs done for Fuzzy", fmt(c.jobs)), tile("hours of GPU time lent", fmt(c.hours)));
    document.getElementById("live-doing").textContent = "Right now Fuzzy is " + (f.doing || "resting") + ".";
    board.replaceChildren(...(d.leaders.length ? d.leaders.map((l, i) => {
      const li = document.createElement("li");
      li.textContent = l.name + " · " + l.hours + " h · " + fmt(l.jobs) + " jobs";
      return li;
    }) : [Object.assign(document.createElement("li"), { textContent: "Nobody yet. Be the first." })]));
    root.hidden = false;
  }
  async function load() {
    try {
      const r = await fetch(API, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      render(await r.json());
      document.getElementById("live-note").textContent = "";
    } catch (e) {
      root.hidden = false;
      document.getElementById("live-note").textContent = "Fuzzy is resting right now, so live numbers are unavailable.";
    }
  }
  load();
  setInterval(() => { if (!document.hidden) load(); }, 15000);
})();
