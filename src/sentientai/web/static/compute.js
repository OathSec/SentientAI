"use strict";
/* The Compute page: the pool of lent computers, and each person's own. */
(async function () {
  const state = await boot();
  if (!state) return;

  const gb = (mb) => (mb ? (mb / 1024).toFixed(0) + " GB" : "");
  function hours(seconds) {
    if (seconds < 90) return Math.round(seconds) + " s";
    if (seconds < 5400) return Math.round(seconds / 60) + " min";
    return (seconds / 3600).toFixed(1) + " h";
  }

  function chip(c) {
    const text = !c.online ? "offline" : c.busy ? "working" : "ready";
    return el("span", { class: "chip " + (!c.online ? "off" : c.busy ? "busy" : "on") }, text);
  }

  function gpuCard(g) {
    const load = g.load_pct === null || g.load_pct === undefined ? null : g.load_pct;
    return el("div", { class: "gpu" },
      el("div", { class: "row spread" }, el("strong", {}, g.name), g.chip || null),
      g.sub ? el("div", { class: "small muted" }, g.sub) : null,
      bar(load === null ? 0 : load / 100),
      el("div", { class: "small muted" },
        (load === null ? "load n/a" : "load " + load + "%") +
        (g.memory_total_mb ? " · " + (g.memory_used_mb ? gb(g.memory_used_mb) + " of " : "") + gb(g.memory_total_mb) : "") +
        (g.temp_c ? " · " + Math.round(g.temp_c) + " °C" : "")));
  }

  // Node.append(array) turns the array into text; always insert the nodes themselves.
  const fill = (node, items) => clear(node).append(...[].concat(items));

  function tile(label, value) { return el("div", { class: "tile" }, el("div", { class: "num" }, String(value)), el("div", { class: "small muted" }, label)); }

  function computerCard(c) {
    const what = c.capabilities.map((x) => ({ llm: "thinking", image: "pictures", embed: "processor" }[x] || x)).join(" + ");
    return el("div", { class: "computer" },
      el("div", { class: "row spread" }, el("strong", {}, c.name), chip(c)),
      c.gpu ? el("div", { class: "small" }, c.gpu + (c.memory_total_mb ? " · " + gb(c.memory_total_mb) : "")) : el("div", { class: "small muted" }, "not connected yet"),
      c.online ? bar((c.load_pct || 0) / 100) : null,
      el("dl", { class: "kv" },
        el("dt", {}, "Sharing"), el("dd", {}, c.online ? (what || "nothing") : "n/a"),
        c.online && c.model ? [el("dt", {}, "Model"), el("dd", {}, c.model + (c.same_model ? " (same as Fuzzy's: helps with conversations)" : " (helps with background thinking)"))] : null,
        el("dt", {}, "Jobs done"), el("dd", {}, c.jobs + (c.jobs ? " (" + c.chat_jobs + " chat, " + c.picture_jobs + " pictures)" : "")),
        el("dt", {}, "Time working"), el("dd", {}, hours(c.busy_s)),
        c.online ? [el("dt", {}, "Connected for"), el("dd", {}, hours(c.connected_s))] : [el("dt", {}, "Last seen"), el("dd", {}, c.last_seen_at ? ago(c.last_seen_at) : "never")]),
      state.admin ? el("button", { class: "link small", onclick: () => setOwn(c) }, c.own ? "Make this a shared computer" : "This is my own computer") : null,
      " ",
      el("button", { class: "link small", onclick: () => unlink(c) }, "Unlink this computer"));
  }

  async function setOwn(c) {
    const own = !c.own;
    if (own && !confirm("Mark " + c.name + " as your own computer? It will be allowed to run Fuzzy's private thinking, including pieces of conversations.")) return;
    try { await api("/api/compute/computers/" + encodeURIComponent(c.id) + "/own", { method: "POST", body: { own } }); } catch (e) { alert(e.message); }
    load();
  }

  async function unlink(c) {
    if (!confirm("Unlink " + c.name + "? It will disconnect and stop counting.")) return;
    try { await api("/api/compute/computers/" + encodeURIComponent(c.id), { method: "DELETE" }); } catch (e) { alert(e.message); }
    load();
  }

  function showLinker(data, downloadUrl) {
    const box = $("linker");
    clear(box).classList.remove("hidden");
    const code = el("input", { type: "text", readonly: true, value: data.code, class: "code" });
    box.append(
      el("h3", {}, "Link " + data.name),
      el("ol", { class: "steps" },
        el("li", {}, downloadUrl ? ["Download the program: ", el("a", { href: downloadUrl }, "Fuzzy Worker for Windows")] : "Get the Fuzzy Worker program for Windows from the person running this site."),
        el("li", {}, "Open it and press ", el("b", {}, "Set everything up"), "."),
        el("li", {}, "Paste this code on the Home tab, press ", el("b", {}, "Link this computer"), ", then ", el("b", {}, "Start"), ".")),
      code,
      el("div", { class: "row" },
        el("button", { onclick: async () => { code.select(); try { await navigator.clipboard.writeText(data.code); } catch (e) { document.execCommand("copy"); } } }, "Copy code"),
        el("button", { class: "link", onclick: () => { box.classList.add("hidden"); load(); } }, "Done")),
      el("p", { class: "small muted" }, "This code is shown once and works only for this computer. Don't share it."));
  }

  $("add").addEventListener("click", async () => {
    const name = prompt("What should we call this computer? (for example “Alex's gaming PC”)");
    if (!name || !name.trim()) return;
    const own = !!(state.admin && confirm("Is this one of YOUR OWN computers? OK lets it run Fuzzy's private thinking (conversations included). Cancel makes it a shared computer that only ever draws public pictures."));
    try {
      const data = await api("/api/compute/computers", { method: "POST", body: { name: name.trim(), own } });
      showLinker(data, lastDownload);
    } catch (e) { alert(e.message); }
  });

  let lastDownload = "";
  async function load() {
    let d;
    try { d = await api("/api/compute"); } catch (e) { clear($("tiles")).append(el("p", { class: "notice error" }, e.message)); return; }
    lastDownload = d.download_url;
    if (d.download_url) $("dl-win").href = d.download_url;
    const p = d.pool;
    clear($("tiles")).append(tile("computers online", p.online), tile("working now", p.busy), tile("jobs done for Fuzzy", p.jobs), tile("contributors", p.contributors), tile("home graphics cards", p.home_gpus.length));

    const t = d.my_totals;
    clear($("mine-totals")).append(
      tile("your computers", t.computers), tile("jobs for Fuzzy", t.jobs), tile("time working", hours(t.busy_s)),
      tile("your rank", t.rank ? "#" + t.rank + " of " + t.of : "n/a"));
    fill($("mine"), d.mine.length ? d.mine.map(computerCard) : el("p", { class: "muted" }, "No computers linked yet. Press “Add a computer” to lend yours."));

    fill($("leaders"), d.leaders.length ? d.leaders.map((l) => el("li", {}, el("span", {}, l.name), el("span", { class: "muted small" }, " " + hours(l.busy_s) + " · " + l.jobs + " jobs"))) : el("li", { class: "muted" }, "Nobody yet. Be the first!"));

    const cards = p.home_gpus.map((g) => gpuCard({ ...g, sub: "Fuzzy's home server", chip: el("span", { class: "chip on" }, "home") }));
    for (const c of d.computers) cards.push(gpuCard({ name: c.gpu || c.name, sub: c.name + " · " + c.owner, load_pct: c.load_pct, memory_total_mb: c.memory_total_mb, chip: chip(c) }));
    fill($("pool"), cards.length ? cards : el("p", { class: "muted" }, "No graphics cards reported."));
  }

  load();
  setInterval(() => { if (!document.hidden) load(); }, 5000);
})();
