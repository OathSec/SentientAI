"use strict";
(async function () {
  const state = await boot("user");
  if (!state) return;
  if (!state.user) { location.href = "/admin"; return; }
  const name = state.name;
  $("title").textContent = "You and " + name;
  const msg = $("msg");

  function say(text, kind) { notice(msg, text, kind); window.scrollTo(0, 0); }
  async function run(work) { try { await work(); } catch (e) { say(e.message, "error"); } }

  async function loadRelationship() {
    const data = await api("/api/me/relationship");
    $("rel-title").textContent = "What " + name + " thinks of you";
    const box = clear($("rel-body"));
    const r = data.relationship;
    if (!r || !r.interactions) { box.append(el("p", { class: "muted" }, "You haven’t talked yet.")); return; }
    box.append(
      el("p", {}, r.opinion || "It hasn’t formed an opinion yet."),
      r.feelings ? el("p", { class: "muted" }, r.feelings) : null,
      el("dl", { class: "kv" },
        el("dt", {}, "Talked"), el("dd", {}, r.interactions + " times since " + clock(r.first_met_at)),
        el("dt", {}, "Last seen"), el("dd", {}, ago(r.last_seen_at)),
        r.nickname ? el("dt", {}, "Calls you") : null, r.nickname ? el("dd", {}, r.nickname) : null,
        el("dt", {}, "Familiarity"), el("dd", {}, bar(r.familiarity)),
        el("dt", {}, "Fondness"), el("dd", {}, bar(r.affinity, { min: -1, max: 1, signed: true })),
        el("dt", {}, "Trust"), el("dd", {}, bar(r.trust))));
    if (data.feelings.length) {
      box.append(el("h3", { class: "mt-s" }, "How it feels about you right now"),
        el("ul", { class: "plain" }, data.feelings.map((f) => el("li", {}, f.emotion, f.cause ? el("span", { class: "muted" }, " – " + f.cause) : null))));
    }
  }

  async function loadInvites() {
    const d = await api("/api/me/invites");
    const box = clear($("invites"));
    const left = d.left === null ? "as many as you like" : d.left + " of " + d.allowed + " left";
    box.append(el("p", { class: "muted" }, "Give a friend a one-time code so they can join. You can invite " + left + "."));
    const shown = el("div", {});
    const make = el("button", { class: "primary", disabled: d.left === 0 ? true : null, onclick: async () => {
      try {
        const r = await api("/api/me/invites", { method: "POST", body: {} });
        const code = el("input", { type: "text", readonly: true, value: r.code, class: "code" });
        clear(shown).append(el("p", {}, "Share this code now; it's shown only once and works for " + r.expires_in_days + " days:"), code,
          el("button", { onclick: async () => { code.select(); try { await navigator.clipboard.writeText(r.code); } catch (e) { document.execCommand("copy"); } } }, "Copy"));
        loadInvites();
      } catch (e) { clear(shown).append(el("p", { class: "notice error" }, e.message)); }
    } }, "Make an invite code");
    box.append(make, shown);
    if (d.invites.length) {
      box.append(el("ul", { class: "plain mt-s" }, d.invites.map((i) => el("li", { class: "small" },
        i.status === "joined" ? (i.joined || "Someone") + " joined" : i.status === "waiting" ? "Waiting for a friend (until " + clock(i.expires_at) + ")" : "Expired"))));
    }
  }

  async function loadNotes() {
    const data = await api("/api/notes");
    const box = clear($("notes"));
    if (!data.notes.length) { box.append(el("p", { class: "muted" }, name + " hasn’t left you any notes.")); return; }
    box.append(el("ul", { class: "plain" }, data.notes.map((n) => el("li", {},
      el("div", { class: "pre" }, n.content),
      el("div", { class: "small muted" }, clock(n.at) + (n.read ? "" : " · unread"))))));
    if (data.notes.some((n) => !n.read)) await api("/api/notes/read", { method: "POST" });
  }

  function loadTotp() {
    const box = clear($("totp"));
    if (state.user.totp_enabled) {
      const form = el("form", {}, el("p", {}, "Two-factor sign-in is on."),
        el("label", { for: "td-pass" }, "Password to turn it off"),
        el("input", { id: "td-pass", type: "password", autocomplete: "current-password", required: true, maxlength: "256" }),
        el("p", { class: "mt-s" }, el("button", { type: "submit", class: "danger" }, "Turn off")));
      form.addEventListener("submit", (e) => { e.preventDefault(); run(async () => {
        await api("/api/me/totp/disable", { method: "POST", body: { password: $("td-pass").value } });
        state.user.totp_enabled = false; say("Two-factor sign-in is off."); loadTotp();
      }); });
      box.append(form);
      return;
    }
    box.append(el("p", {}, "Add a second step to signing in, using an authenticator app."),
      el("button", { class: "secondary", onclick: () => run(async () => {
        const begin = await api("/api/me/totp/begin", { method: "POST" });
        clear(box);
        const form = el("form", {},
          el("p", {}, "In your authenticator app, add an account by entering this key:"),
          el("p", {}, el("code", { class: "wrap" }, begin.secret)),
          el("p", { class: "small muted" }, "Or open this address on your phone: "), el("p", { class: "small wrap" }, begin.uri),
          el("label", { for: "te-code" }, "Then enter the 6-digit code it shows"),
          el("input", { id: "te-code", type: "text", inputmode: "numeric", autocomplete: "one-time-code", maxlength: "8", required: true }),
          el("p", { class: "mt-s" }, el("button", { type: "submit" }, "Turn on")));
        form.addEventListener("submit", (e) => { e.preventDefault(); run(async () => {
          await api("/api/me/totp/enable", { method: "POST", body: { secret: begin.secret, code: $("te-code").value } });
          state.user.totp_enabled = true; say("Two-factor sign-in is on."); loadTotp();
        }); });
        box.append(form);
      }) }, "Set up"));
  }

  $("pw").addEventListener("submit", (e) => { e.preventDefault(); run(async () => {
    await api("/api/me/password", { method: "POST", body: { current: $("pw-cur").value, new: $("pw-new").value } });
    $("pw").reset(); say("Password changed. Other devices have been signed out.");
  }); });
  $("erase").addEventListener("submit", (e) => { e.preventDefault(); run(async () => {
    if (!confirm("Delete your account and everything private to you? This can't be undone.")) return;
    await api("/api/me/delete", { method: "POST", body: { password: $("er-pass").value } });
    location.href = "/login";
  }); });

  const me = await api("/api/me");
  const lim = me.limits;
  $("usage").textContent = "Today: " + me.usage.messages + (lim.messages ? " of " + lim.messages : "") + " messages" +
    (me.image_generation ? ", " + me.usage.images + (lim.images ? " of " + lim.images : "") + " images." : ".");
  await Promise.all([loadRelationship(), loadNotes(), loadInvites()]);
  loadTotp();
})();
