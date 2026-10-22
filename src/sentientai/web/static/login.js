"use strict";
(async function () {
  let state;
  try { state = await api("/api/auth/state"); } catch (e) { notice($("msg"), e.message, "error"); return; }
  App.csrf = state.csrf;
  document.title = "Sign in – " + state.name;
  buildHeader(state);
  $("heading").textContent = state.name;

  const forms = { signin: $("signin"), register: $("register"), second: $("second") };
  function show(name) {
    for (const [key, form] of Object.entries(forms)) form.classList.toggle("hidden", key !== name);
    notice($("msg"), "");
  }

  if (state.signed_in && !state.mfa_pending) { location.href = safeNext(); return; }
  if (state.mfa_pending) show("second");
  if (state.registration !== "closed") $("to-register").classList.remove("hidden");
  if (state.registration !== "invite") $("invite-row").classList.add("hidden");
  $("show-register").addEventListener("click", (e) => { e.preventDefault(); show("register"); });
  $("show-signin").addEventListener("click", (e) => { e.preventDefault(); show("signin"); });

  async function submit(form, work) {
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try { await work(); } catch (e) { notice($("msg"), e.message, "error"); } finally { button.disabled = false; }
  }
  function done(result) {
    App.csrf = result.csrf;
    if (result.mfa_required) show("second"); else location.href = safeNext();
  }

  forms.signin.addEventListener("submit", (e) => {
    e.preventDefault();
    submit(forms.signin, async () => {
      done(await api("/api/auth/login", { method: "POST", body: { username: $("si-user").value, password: $("si-pass").value } }));
    });
  });
  forms.register.addEventListener("submit", (e) => {
    e.preventDefault();
    submit(forms.register, async () => {
      done(await api("/api/auth/register", {
        method: "POST",
        body: {
          username: $("rg-user").value,
          password: $("rg-pass").value,
          display_name: $("rg-name").value || null,
          invite_code: $("rg-invite").value || null,
        },
      }));
    });
  });
  forms.second.addEventListener("submit", (e) => {
    e.preventDefault();
    submit(forms.second, async () => {
      await api("/api/auth/mfa", { method: "POST", body: { code: $("code").value } });
      location.href = safeNext();
    });
  });
})();
