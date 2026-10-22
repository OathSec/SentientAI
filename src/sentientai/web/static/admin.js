"use strict";
(async function () {
  const state = await boot("admin");
  if (!state) return;
  const msg = $("msg"), panel = $("panel");
  const me = state.user ? state.user.id : null;
  function say(text, kind) { notice(msg, text, kind); window.scrollTo(0, 0); }
  async function act(work) { try { await work(); } catch (e) { say(e.message, "error"); } }

  const TABS = { users: "People", invites: "Invites", safety: "Safety", audit: "Audit log", memory: "Memory", tasks: "Tasks" };
  const renderers = { users, invites, safety, audit, memory, tasks };
  let current = "users", poll = null;

  function tabs() {
    const bar = clear($("tabs"));
    for (const [key, label] of Object.entries(TABS)) {
      bar.append(el("button", { class: key === current ? "" : "secondary", role: "tab", "aria-selected": key === current, onclick: () => open(key) }, label));
    }
  }
  function open(key) { current = key; tabs(); notice(msg, ""); if (poll) clearInterval(poll); poll = null; renderers[key](); }

  async function users() {
    const data = await api("/api/admin/users");
    clear(panel).append(el("table", {},
      el("thead", {}, el("tr", {}, ["Person", "Role", "Status", "Talked", "Today", "Last sign-in", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, data.users.map((u) => el("tr", {},
        el("td", {}, el("strong", {}, u.display_name), el("div", { class: "small muted" }, u.username + (u.totp_enabled ? " · 2FA" : "")),
          u.opinion ? el("div", { class: "small" }, u.opinion) : null),
        el("td", {}, u.role, u.can_teach ? el("span", { class: "tag" }, "teaches") : null),
        el("td", {}, u.status, u.locked ? el("span", { class: "tag" }, "locked") : null),
        el("td", {}, u.interactions),
        el("td", {}, u.today.messages + " msg, " + u.today.images + " img"),
        el("td", {}, u.last_login_at ? ago(u.last_login_at) : "never"),
        el("td", {}, u.id === me ? el("span", { class: "muted small" }, "you") : el("div", { class: "row" },
          el("button", { class: "secondary", onclick: () => act(async () => { await api("/api/admin/users/" + u.id, { method: "POST", body: { status: u.status === "active" ? "disabled" : "active" } }); users(); }) }, u.status === "active" ? "Disable" : "Enable"),
          el("button", { class: "secondary", onclick: () => act(async () => { await api("/api/admin/users/" + u.id, { method: "POST", body: { can_teach: !u.can_teach } }); users(); }) }, u.can_teach ? "Stop teaching" : "Let teach"),
          el("button", { class: "secondary", onclick: () => act(async () => {
            if (!confirm("Make " + u.display_name + (u.role === "admin" ? " a regular user?" : " an administrator?"))) return;
            await api("/api/admin/users/" + u.id, { method: "POST", body: { role: u.role === "admin" ? "user" : "admin" } }); users();
          }) }, u.role === "admin" ? "Demote" : "Make admin"),
          el("button", { class: "danger", onclick: () => act(async () => {
            if (!confirm("Erase " + u.display_name + " and everything private to them? This can't be undone.")) return;
            await api("/api/admin/users/" + u.id, { method: "DELETE" }); users();
          }) }, "Erase")))))))));
  }

  async function invites() {
    const data = await api("/api/admin/invites");
    const form = el("form", { class: "card" }, el("h2", {}, "New invite"),
      el("label", { for: "iv-role" }, "Role"),
      el("select", { id: "iv-role" }, el("option", { value: "user" }, "User"), el("option", { value: "admin" }, "Administrator")),
      el("label", { class: "row small mt-xs" }, el("input", { type: "checkbox", id: "iv-teach" }), "Can teach Fuzzy"),
      el("label", { for: "iv-days" }, "Valid for (days)"),
      el("input", { id: "iv-days", type: "number", min: "1", max: "90", value: "7" }),
      el("p", { class: "mt-s" }, el("button", { type: "submit" }, "Create invite")),
      el("div", { id: "iv-out" }));
    form.addEventListener("submit", (e) => { e.preventDefault(); act(async () => {
      const r = await api("/api/admin/invites", { method: "POST", body: { role: $("iv-role").value, can_teach: $("iv-teach").checked, days: Number($("iv-days").value) } });
      clear($("iv-out")).append(el("p", { class: "notice" }, "Give them this code. It is shown once and works once: ", el("code", {}, r.code)));
      invites();
    }); });
    clear(panel).append(form, el("div", { class: "card" }, el("h2", {}, "Invites"),
      data.invites.length ? el("table", {}, el("thead", {}, el("tr", {}, ["Role", "Teaches", "Created", "Expires", "Used"].map((h) => el("th", {}, h)))),
        el("tbody", {}, data.invites.map((i) => el("tr", {}, el("td", {}, i.role), el("td", {}, i.can_teach ? "yes" : "no"), el("td", {}, clock(i.created_at)), el("td", {}, clock(i.expires_at)), el("td", {}, i.used ? "yes" : "no")))))
        : el("p", { class: "muted" }, "None yet.")));
  }

  async function safety() {
    const [st, approvals] = await Promise.all([api("/v1/state"), api("/v1/approvals")]);
    const kill = st.kill_switch;
    const box = el("div", {},
      el("div", { class: "card" }, el("h2", {}, "Kill switch"),
        el("p", {}, kill.engaged ? "ENGAGED: " + (kill.reason || "") + ". " + state.name + " is not acting." : "Released. " + state.name + " is free to act."),
        kill.engaged
          ? el("button", { onclick: () => act(async () => { await api("/v1/resume", { method: "POST" }); safety(); }) }, "Resume")
          : el("button", { class: "danger", onclick: () => act(async () => { await api("/v1/kill", { method: "POST", body: { reason: "stopped from the admin page" } }); safety(); }) }, "Stop everything now")),
      el("div", { class: "card" }, el("h2", {}, "Waiting for your approval"),
        approvals.length ? el("ul", { class: "plain" }, approvals.map((a) => el("li", {},
          el("strong", {}, a.tool), " ", el("span", { class: "tag" }, a.risk), el("div", { class: "small wrap" }, JSON.stringify(a.arguments)),
          el("div", { class: "small muted" }, (a.reasons || []).join("; ")),
          el("div", { class: "row" },
            el("button", { onclick: () => act(async () => { await api("/v1/approvals/" + a.id, { method: "POST", body: { approve: true } }); safety(); }) }, "Approve"),
            el("button", { class: "danger", onclick: () => act(async () => { await api("/v1/approvals/" + a.id, { method: "POST", body: { approve: false } }); safety(); }) }, "Deny")))))
          : el("p", { class: "muted" }, "Nothing waiting.")));
    clear(panel).append(box);
    poll = setInterval(() => { if (current === "safety") safety(); }, 10000);
  }

  async function audit() {
    const [tail, verify] = await Promise.all([api("/v1/audit/tail?n=150"), api("/v1/audit/verify")]);
    clear(panel).append(
      el("p", { class: verify.ok ? "muted" : "notice error" }, verify.ok ? "Audit chain intact (" + verify.entries + " entries)." : "AUDIT CHAIN BROKEN: " + verify.error),
      el("div", { class: "card" }, el("pre", { class: "feed tall" },
        tail.slice().reverse().map((e) => e.ts.slice(0, 19).replace("T", " ") + "  " + e.type + "  " + JSON.stringify(e.payload).slice(0, 260)).join("\n"))));
  }

  async function memory() {
    const rows = await api("/api/admin/memories?limit=100");
    clear(panel).append(el("div", { class: "card" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Kind", "Memory", "Whose", "When", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((m) => el("tr", {},
        el("td", {}, m.kind), el("td", { class: "pre wrap" }, m.content.slice(0, 600)),
        el("td", {}, m.user_id ? "private" : "shared"), el("td", {}, ago(m.at)),
        el("td", {}, el("button", { class: "danger", onclick: () => act(async () => { await api("/v1/memory/" + m.id + "?hard=true", { method: "DELETE" }); memory(); }) }, "Forget"))))))));
  }

  async function tasks() {
    const rows = await api("/v1/tasks?limit=60");
    clear(panel).append(el("div", { class: "card" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Status", "Source", "Task", "When", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((t) => el("tr", {},
        el("td", {}, t.status), el("td", {}, t.source),
        el("td", { class: "wrap" }, t.user_id ? "(a conversation)" : t.goal.slice(0, 200)),
        el("td", {}, ago(t.created_at)),
        el("td", {}, ["pending", "running"].includes(t.status) ? el("button", { class: "secondary", onclick: () => act(async () => { await api("/v1/tasks/" + t.id + "/cancel", { method: "POST" }); tasks(); }) }, "Cancel") : null)))))));
  }

  open("users");
})();
