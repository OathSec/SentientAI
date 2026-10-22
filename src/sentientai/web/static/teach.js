"use strict";
(async function () {
  const state = await boot("teacher");
  if (!state) return;
  const msg = $("msg");
  function say(text, kind) { notice(msg, text, kind); window.scrollTo(0, 0); }
  async function run(form, work) {
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try { await work(); await loadDocs(); } catch (e) { say(e.message, "error"); } finally { button.disabled = false; }
  }
  const done = (r) => say(r.already_known ? "Fuzzy already knew “" + r.title + "”." : "Taught: “" + r.title + "” (" + r.chunks + " pieces).");

  $("text-form").addEventListener("submit", (e) => { e.preventDefault(); run($("text-form"), async () => {
    done(await api("/api/teach/text", { method: "POST", body: { title: $("t-title").value, text: $("t-text").value } }));
    $("text-form").reset();
  }); });
  $("url-form").addEventListener("submit", (e) => { e.preventDefault(); run($("url-form"), async () => {
    done(await api("/api/teach/url", { method: "POST", body: { url: $("u-url").value } }));
    $("url-form").reset();
  }); });
  $("file-form").addEventListener("submit", (e) => { e.preventDefault(); run($("file-form"), async () => {
    const file = $("f-file").files[0];
    if (!file) return;
    done(await api("/api/teach/file?name=" + encodeURIComponent(file.name), { method: "POST", raw: await file.arrayBuffer() }));
    $("file-form").reset();
  }); });

  async function loadDocs() {
    const data = await api("/api/teach/documents");
    const box = clear($("docs"));
    if (!data.documents.length) { box.append(el("p", { class: "muted" }, "Nothing taught yet.")); return; }
    box.append(el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Title"), el("th", {}, "Pieces"), el("th", {}, "Added"), el("th", {}, "Read by Fuzzy"), state.admin ? el("th") : null)),
      el("tbody", {}, data.documents.map((d) => el("tr", {},
        el("td", {}, d.title), el("td", {}, d.chunks), el("td", {}, ago(d.at)), el("td", {}, d.read ? "yes" : "not yet"),
        state.admin ? el("td", {}, el("button", { class: "danger", onclick: async () => {
          if (!confirm("Make Fuzzy forget “" + d.title + "”?")) return;
          try { await api("/api/admin/documents/" + d.id, { method: "DELETE" }); loadDocs(); } catch (e) { say(e.message, "error"); }
        } }, "Forget")) : null)))));
  }
  await loadDocs();
})();
