"use strict";
/* Server hardware panel (administrators): GPUs, CPU, memory, temperatures, disks. */
const Hardware = (function () {
  const gb = (mb) => (mb / 1024).toFixed(1) + " GB";
  const temp = (c) => (c === null || c === undefined ? "n/a" : c.toFixed(0) + " °C");
  function hot(c, warn, crit) { return c === null || c === undefined ? "" : c >= crit ? " warn" : c >= warn ? " hot" : ""; }

  function render(box, d) {
    clear(box);
    if (!d.gpus.length) {
      box.append(el("p", { class: "muted" }, "No GPU readings. nvidia-smi isn't available to this container (the server may need the NVIDIA runtime's “utility” capability)."));
    }
    for (const g of d.gpus) {
      box.append(el("div", { class: "hw" },
        el("div", { class: "row spread" }, el("strong", {}, "GPU " + g.index + " · " + g.name), el("span", { class: "temp" + hot(g.temp_c, 75, 85) }, temp(g.temp_c))),
        bar(g.util_pct === null ? 0 : g.util_pct / 100),
        el("div", { class: "small muted" },
          "load " + (g.util_pct === null ? "n/a" : g.util_pct + "%") +
          " · memory " + (g.mem_used_mb === null ? "n/a" : gb(g.mem_used_mb) + " of " + gb(g.mem_total_mb)) +
          (g.power_w === null ? "" : " · " + g.power_w.toFixed(0) + (g.power_limit_w ? " / " + g.power_limit_w.toFixed(0) : "") + " W") +
          (g.fan_pct === null ? "" : " · fan " + g.fan_pct + "%") +
          (g.clock_mhz === null ? "" : " · " + g.clock_mhz.toFixed(0) + " MHz"))));
    }
    const m = d.memory, load = d.cpu.load;
    box.append(el("dl", { class: "kv mt-xs" },
      el("dt", {}, "CPU"), el("dd", {}, d.cpu.model + " · " + d.cpu.threads + " threads" + (load.length ? " · load " + load.join(" / ") : "")),
      el("dt", {}, "Memory"), el("dd", {}, m.total_mb ? gb(m.used_mb) + " used of " + gb(m.total_mb) : "n/a"),
      el("dt", {}, "Uptime"), el("dd", {}, d.uptime_s ? duration(d.uptime_s) : "n/a")));
    if (d.temperatures.length) {
      box.append(el("h3", { class: "mt-xs" }, "Temperatures"),
        el("ul", { class: "plain" }, d.temperatures.map((t) => el("li", { class: "row spread small" }, t.sensor, el("span", { class: "temp" + hot(t.temp_c, 80, 92) }, temp(t.temp_c))))));
    } else box.append(el("p", { class: "small muted" }, "No CPU/board temperature sensors are visible from inside the container."));
    if (d.disks.length) {
      box.append(el("h3", { class: "mt-xs" }, "Disks"),
        el("ul", { class: "plain" }, d.disks.map((x) => el("li", { class: "small" }, x.path + ": " + x.used_gb + " of " + x.total_gb + " GB used, " + x.free_gb + " GB free"))));
    }
  }

  function mount(box, every) {
    async function load() {
      try { render(box, await api("/api/admin/hardware")); }
      catch (e) { clear(box).append(el("p", { class: "notice error" }, e.message)); }
    }
    load();
    return setInterval(() => { if (!document.hidden && box.offsetParent !== null) load(); }, every || 5000);
  }
  return { mount, render };
})();
