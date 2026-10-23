"use strict";
/* Shared helpers. Everything dynamic is inserted with textContent: nothing here parses HTML. */

const App = { state: null, csrf: null };

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "style") throw new Error("inline styles are blocked by the CSP; use a class");
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
function $(id) { return document.getElementById(id); }

class ApiError extends Error {
  constructor(status, detail) { super(typeof detail === "string" ? detail : "Something went wrong."); this.status = status; }
}

async function api(path, { method = "GET", body, raw, headers = {} } = {}) {
  const init = { method, headers: { ...headers }, credentials: "same-origin" };
  if (method !== "GET" && App.csrf) init.headers["X-CSRF-Token"] = App.csrf;
  if (raw !== undefined) {
    init.body = raw;
    init.headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["Content-Type"] = "application/json";
  } else if (method !== "GET") {
    init.headers["Content-Type"] = "application/json";
    init.body = "{}";
  }
  let response;
  try { response = await fetch(path, init); } catch (e) { throw new ApiError(0, "Can't reach the server."); }
  let data = null;
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) data = await response.json();
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    throw new ApiError(response.status, detail || response.statusText);
  }
  return data;
}

function ago(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}
function duration(seconds) {
  if (seconds < 90) return Math.round(seconds) + " seconds";
  if (seconds < 5400) return Math.round(seconds / 60) + " minutes";
  if (seconds < 129600) return (seconds / 3600).toFixed(1) + " hours";
  return (seconds / 86400).toFixed(1) + " days";
}
function clock(iso) { return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); }

const NAV = [
  ["/", "Chat", (s) => s.signed_in],
  ["/brain", "Brain", (s) => s.signed_in || s.public_pages],
  ["/world", "World", (s) => s.signed_in || s.public_pages],
  ["/gallery", "Gallery", (s) => s.signed_in || s.public_pages],
  ["/compute", "Compute", (s) => s.signed_in],
  ["/teach", "Teach", (s) => s.signed_in && s.user && s.user.can_teach],
  ["/you", "You", (s) => s.signed_in],
  ["/admin", "Admin", (s) => s.admin],
];

function buildHeader(state) {
  const header = $("top");
  if (!header) return;
  clear(header);
  header.append(el("a", { class: "brand", href: "/" }, state.name));
  const nav = el("nav");
  for (const [href, label, show] of NAV) {
    if (!show(state)) continue;
    const link = el("a", { href, "aria-current": location.pathname === href ? "page" : null }, label);
    if (href === "/admin") link.append(el("span", { class: "badge hidden", id: "approvals-badge", title: "Waiting for your approval" }));
    nav.append(link);
  }
  header.append(nav);
  if (state.signed_in && state.user) {
    header.append(el("span", { class: "who" }, state.user.display_name + " "));
    header.append(el("button", { class: "link", onclick: signOut }, "Sign out"));
  } else {
    header.append(el("a", { href: "/login?next=" + encodeURIComponent(location.pathname) }, "Sign in"));
  }
}

async function signOut() {
  try { await api("/api/auth/logout", { method: "POST" }); } catch (e) { /* already signed out */ }
  location.href = "/login";
}

/* Administrators: a count on the Admin link while Fuzzy waits for an approval. */
function showApprovals(count) {
  const badge = $("approvals-badge");
  if (!badge) return;
  badge.textContent = count ? String(count) : "";
  badge.classList.toggle("hidden", !count);
}
function watchApprovals(state) {
  if (!state.admin) return;
  showApprovals(state.pending_approvals);
  setInterval(async () => {
    if (document.hidden) return;
    try { showApprovals((await api("/api/auth/state")).pending_approvals); } catch (e) { /* next time */ }
  }, 20000);
}

/* require: "user" (must be signed in), "any" (public pages allowed), "admin", "teacher" */
async function boot(require) {
  let state;
  try { state = await api("/api/auth/state"); } catch (e) {
    document.body.append(el("main", { class: "narrow" }, el("p", { class: "notice error" }, e.message)));
    return null;
  }
  App.state = state;
  App.csrf = state.csrf;
  document.title = state.name + (document.title ? " – " + document.title : "");
  const next = "/login?next=" + encodeURIComponent(location.pathname);
  if (state.mfa_pending) { location.href = next; return null; }
  if (require !== "any" && !state.signed_in) { location.href = next; return null; }
  if (require === "any" && !state.signed_in && !state.public_pages) { location.href = next; return null; }
  if (require === "admin" && !state.admin) { location.href = "/"; return null; }
  if (require === "teacher" && !(state.user && state.user.can_teach) && !state.admin) { location.href = "/"; return null; }
  buildHeader(state);
  watchApprovals(state);
  return state;
}

function notice(container, message, kind) {
  clear(container);
  if (message) container.append(el("div", { class: "notice" + (kind === "error" ? " error" : "") }, message));
}

function bar(value, { min = 0, max = 1, signed = false } = {}) {
  const fraction = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const cls = signed ? (value > 0.05 ? "meter pos" : value < -0.05 ? "meter neg" : "meter") : "meter";
  const inner = el("span");
  inner.style.width = (fraction * 100).toFixed(0) + "%";
  return el("div", { class: cls, role: "meter", "aria-valuenow": value.toFixed(2) }, inner);
}

function safeNext() {
  const next = new URLSearchParams(location.search).get("next") || "/";
  return next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : "/";
}

/* Live events (Server-Sent Events). Reconnects with a pause; stops when the page is hidden. */
function live(path, onEvent) {
  let source = null, closed = false;
  function open() {
    if (closed) return;
    source = new EventSource(path, { withCredentials: true });
    source.onmessage = (message) => { try { onEvent(JSON.parse(message.data)); } catch (e) { /* ignore */ } };
    source.onerror = () => { source.close(); if (!closed) setTimeout(open, 5000); };
  }
  open();
  return () => { closed = true; if (source) source.close(); };
}

/* A small, safe Markdown renderer for Fuzzy's replies. It builds DOM nodes with textContent
   only (never innerHTML), so nothing the model or a person writes can run as script or markup. */
const Markdown = (function () {
  const SAFE_URL = /^(https?:\/\/|mailto:|\/media\/)/i;
  const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*|__[^_\n]+?__)|(\*[^*\s][^*\n]*?\*|(?<![\w])_[^_\s][^_\n]*?_(?![\w]))|(!?\[[^\]\n]*\]\([^)\s]+\))|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g;

  function inline(text) {
    const out = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(INLINE)) {
      if (m.index > last) out.append(document.createTextNode(text.slice(last, m.index)));
      const token = m[0];
      if (m[1]) out.append(el("code", {}, token.slice(1, -1)));
      else if (m[2]) out.append(el("strong", {}, inline(token.slice(2, -2))));
      else if (m[3]) out.append(el("em", {}, inline(token.slice(1, -1))));
      else if (m[4]) {
        const parts = /^(!?)\[([^\]]*)\]\(([^)\s]+)\)$/.exec(token);
        const url = parts[3];
        if (parts[1] && url.startsWith("/media/")) out.append(el("img", { src: url, alt: parts[2], loading: "lazy" }));
        else if (SAFE_URL.test(url)) out.append(el("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, parts[2] || url));
        else out.append(document.createTextNode(token));
      } else out.append(el("a", { href: token, target: "_blank", rel: "noopener noreferrer" }, token));
      last = m.index + token.length;
    }
    if (last < text.length) out.append(document.createTextNode(text.slice(last)));
    return out;
  }

  function lines(text, container) {
    const parts = text.split("\n");
    parts.forEach((line, i) => { if (i) container.append(el("br")); container.append(inline(line)); });
    return container;
  }

  function render(source) {
    const root = el("div", { class: "md" });
    const rows = String(source || "").replace(/\r\n?/g, "\n").split("\n");
    let i = 0;
    const isBlock = (l) => /^(```|#{1,6}\s|>\s?|\s*[-*•]\s+|\s*\d+[.)]\s+|\s*([-*_])\2{2,}\s*$)/.test(l);
    while (i < rows.length) {
      const line = rows[i];
      if (!line.trim()) { i++; continue; }
      const fence = /^```\s*([\w+-]*)\s*$/.exec(line);
      if (fence) {
        const body = [];
        i++;
        while (i < rows.length && !/^```\s*$/.test(rows[i])) body.push(rows[i++]);
        i++;
        const pre = el("pre", {}, el("code", {}, body.join("\n")));
        if (fence[1]) pre.dataset.lang = fence[1];
        root.append(pre);
        continue;
      }
      const heading = /^(#{1,6})\s+(.*)$/.exec(line);
      if (heading) { root.append(el("p", { class: "md-h" }, inline(heading[2]))); i++; continue; }
      if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { root.append(el("hr")); i++; continue; }
      if (/^>\s?/.test(line)) {
        const quote = [];
        while (i < rows.length && /^>\s?/.test(rows[i])) quote.push(rows[i++].replace(/^>\s?/, ""));
        root.append(lines(quote.join("\n"), el("blockquote")));
        continue;
      }
      const bullet = /^\s*[-*•]\s+/, numbered = /^\s*\d+[.)]\s+/;
      if (bullet.test(line) || numbered.test(line)) {
        const ordered = numbered.test(line);
        const list = el(ordered ? "ol" : "ul");
        const marker = ordered ? numbered : bullet;
        while (i < rows.length && marker.test(rows[i])) list.append(el("li", {}, inline(rows[i++].replace(marker, ""))));
        root.append(list);
        continue;
      }
      const para = [line];
      i++;
      while (i < rows.length && rows[i].trim() && !isBlock(rows[i])) para.push(rows[i++]);
      root.append(lines(para.join("\n"), el("p")));
    }
    return root;
  }
  return { render };
})();
