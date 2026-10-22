"use strict";
(async function () {
  const state = await boot("any");
  if (!state) return;
  $("title").textContent = state.name + "’s gallery";
  try {
    const data = await api("/api/gallery");
    $("intro").textContent = state.signed_in
      ? "Pictures " + state.name + " has made, and any it made for you. Ask for one in the chat."
      : "Pictures " + state.name + " has made.";
    const box = clear($("items"));
    if (!data.images.length) { $("intro").textContent += " Nothing here yet."; return; }
    for (const image of data.images) {
      box.append(el("figure", {},
        el("a", { href: image.url }, el("img", { src: image.url, alt: image.caption || "An image", loading: "lazy" })),
        el("figcaption", {}, image.mine ? el("span", { class: "tag" }, "for you") : null, image.caption || "", " ", el("span", { class: "muted" }, ago(image.at)))));
    }
  } catch (e) { $("intro").textContent = e.message; }
})();
