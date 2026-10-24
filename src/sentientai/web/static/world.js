"use strict";
(async function () {
  const state = await boot("any");
  if (!state) return;
  const name = state.name;
  const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  let world;
  try { world = await api("/api/world"); } catch (e) { $("now").textContent = e.message; return; }

  // ---------------------------------------------------------------- scene
  const canvas = $("canvas"), stage = $("stage");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(dark ? 0x1d2420 : 0xe6eee8);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);

  scene.add(new THREE.HemisphereLight(0xffffff, dark ? 0x333a35 : 0x9aa89d, 0.5));
  const sun = new THREE.DirectionalLight(0xffffff, 0.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshLambertMaterial({ color: dark ? 0x3c4a3c : 0xb9c8a8 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1)),
    new THREE.LineBasicMaterial({ color: dark ? 0x68786a : 0x8a9a80 }));
  edge.rotation.x = -Math.PI / 2; edge.position.y = 0.01;
  scene.add(edge);
  // The ground is Fuzzy's to enlarge, so its size can change while the page is open.
  function setSize(size) {
    world.size = size;
    const half = size / 2;
    ground.scale.set(size, size, 1);
    edge.scale.set(size, size, 1);
    sun.position.set(half * 0.6, half * 1.2, half * 0.4);
    Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: half * 4 });
    sun.shadow.camera.updateProjectionMatrix();
    camera.far = Math.max(500, size * 4); camera.updateProjectionMatrix();
  }
  setSize(world.size);

  // ---------------------------------------------------------------- shapes
  const rad = (d) => (d * Math.PI) / 180;
  function geometry(p) {
    switch (p.shape) {
      case "sphere": return new THREE.SphereGeometry(p.radius, 24, 16);
      case "cylinder": return new THREE.CylinderGeometry(p.radius, p.radius, p.height, 24);
      case "cone": return new THREE.ConeGeometry(p.radius, p.height, 24);
      case "torus": return new THREE.TorusGeometry(p.radius, p.tube, 12, 32);
      default: return new THREE.BoxGeometry(p.size[0], p.size[1], p.size[2]);
    }
  }
  // Fuzzy's own paintings, shown on "picture" parts. Each image is loaded once and shared.
  const textures = new Map();
  function texture(id) {
    if (!textures.has(id)) textures.set(id, new THREE.TextureLoader().load("/media/" + encodeURIComponent(id)));
    return textures.get(id);
  }
  function makeMesh(p) {
    if (p.shape === "picture" && p.media_id) {
      const frame = new THREE.MeshLambertMaterial({ color: p.color });
      const art = new THREE.MeshLambertMaterial({ map: texture(p.media_id) });
      // Box faces: +x, -x, +y, -y, +z (the picture), -z.
      return new THREE.Mesh(new THREE.BoxGeometry(p.size[0], p.size[1], Math.max(0.02, p.size[2])), [frame, frame, frame, frame, art, frame]);
    }
    return new THREE.Mesh(geometry(p), new THREE.MeshLambertMaterial({ color: p.color }));
  }
  function build(parts) {
    const group = new THREE.Group();
    for (const p of parts) {
      const mesh = makeMesh(p);
      mesh.position.set(p.position[0], p.position[1], p.position[2]);
      mesh.rotation.set(rad(p.rotation[0]), rad(p.rotation[1]), rad(p.rotation[2]));
      mesh.castShadow = true; mesh.receiveShadow = true;
      group.add(mesh);
    }
    // Models place parts around the centre; never let a thing sink into the ground.
    const box = new THREE.Box3().setFromObject(group);
    if (isFinite(box.min.y) && box.min.y < -0.01) group.children.forEach((m) => { m.position.y -= box.min.y; });
    return group;
  }
  function dispose(node) {
    // Shared picture textures are kept (another object may show the same painting).
    node.traverse((n) => {
      if (n.geometry) n.geometry.dispose();
      for (const m of [].concat(n.material || [])) m.dispose();
    });
  }

  // --------------------------------------------------------------- objects
  const objects = new Map();
  function place(obj, animate) {
    const old = objects.get(obj.id);
    if (old) { scene.remove(old.node); dispose(old.node); }
    const inner = build(obj.parts);
    const node = new THREE.Group();
    node.add(inner);
    node.position.set(obj.x, 0, obj.z);
    node.rotation.y = rad(obj.rotation || 0);
    node.userData.id = obj.id;
    scene.add(node);
    objects.set(obj.id, { node, data: obj, born: animate ? performance.now() : null });
    if (animate) node.scale.set(0.01, 0.01, 0.01);
  }
  function remove(id) {
    const entry = objects.get(id);
    if (!entry) return;
    scene.remove(entry.node); dispose(entry.node); objects.delete(id);
  }

  // ---------------------------------------------------------------- avatar
  let avatar = null, avatarData = world.avatar;
  const shown = { x: world.body.x, z: world.body.z, heading: world.body.heading || 0 };
  function buildAvatar() {
    if (avatar) { scene.remove(avatar); dispose(avatar); }
    avatar = new THREE.Group();
    if (avatarData && avatarData.parts && avatarData.parts.length) avatar.add(build(avatarData.parts));
    else {
      const colour = dark ? 0x6fb59b : 0x2f6b57;
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 1, 16), new THREE.MeshLambertMaterial({ color: colour }));
      body.position.y = 0.5; body.castShadow = true;
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 12), new THREE.MeshLambertMaterial({ color: colour }));
      head.position.y = 1.2; head.castShadow = true;
      avatar.add(body, head);
    }
    scene.add(avatar);
  }
  buildAvatar();
  let body = world.body;

  // ---------------------------------------------------------------- camera
  const view = { azimuth: 0.6, polar: 0.95, distance: Math.max(8, Math.min(60, world.size * 0.55)), target: new THREE.Vector3(0, 1, 0), focus: null };
  const maxDistance = () => Math.max(60, world.size * 1.6);
  function updateCamera() {
    const t = view.focus || (follow.checked ? new THREE.Vector3(shown.x, 1, shown.z) : null);
    if (t) view.target.lerp(t, 0.08);
    const sp = Math.sin(view.polar), cp = Math.cos(view.polar);
    camera.position.set(
      view.target.x + view.distance * sp * Math.sin(view.azimuth),
      view.target.y + view.distance * cp,
      view.target.z + view.distance * sp * Math.cos(view.azimuth));
    camera.lookAt(view.target);
    camera.far = Math.max(500, world.size * 4, view.distance * 4);
    camera.updateProjectionMatrix();
  }
  const follow = $("follow");
  follow.addEventListener("change", () => { if (follow.checked) view.focus = null; });

  // The side panel can be tucked away (remembered; starts tucked away on a phone).
  const side = $("side"), sideToggle = $("sidetoggle");
  function setSide(open) {
    side.classList.toggle("collapsed", !open);
    sideToggle.textContent = open ? "Hide panel" : "Show panel";
    sideToggle.setAttribute("aria-expanded", String(open));
  }
  let saved = null;
  try { saved = localStorage.getItem("fuzzy.world.panel"); } catch (e) { /* storage may be blocked */ }
  setSide(saved ? saved === "open" : !window.matchMedia("(max-width: 40rem)").matches);
  sideToggle.addEventListener("click", () => {
    const open = side.classList.contains("collapsed");
    setSide(open);
    try { localStorage.setItem("fuzzy.world.panel", open ? "open" : "closed"); } catch (e) { /* ignore */ }
  });

  // Moving the view yourself (panning) stops following Fuzzy; tick "Follow" to go back to it.
  function freeLook() { follow.checked = false; view.focus = null; }
  function pan(right, forward) {
    freeLook();
    const s = Math.sin(view.azimuth), c = Math.cos(view.azimuth);
    const limit = world.size / 2 + 10;
    view.target.x = Math.max(-limit, Math.min(limit, view.target.x + right * c - forward * s));
    view.target.z = Math.max(-limit, Math.min(limit, view.target.z - right * s - forward * c));
  }
  function zoom(factor) { view.distance = Math.max(1.5, Math.min(maxDistance(), view.distance * factor)); }

  // Mouse: drag to look around, right-drag (or Shift+drag) to move. Touch: one finger looks,
  // two fingers move and pinch to zoom. Dragging works from anywhere on the stage, including over
  // the thought bubble, but not over the side panel.
  const pointers = new Map();
  let moved = 0, panning = false, pinch = 0;
  stage.addEventListener("contextmenu", (e) => { if (!$("side").contains(e.target)) e.preventDefault(); });
  stage.addEventListener("pointerdown", (e) => {
    if ($("side").contains(e.target)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved = 0;
    panning = e.button === 2 || e.shiftKey;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = Math.hypot(a.x - b.x, a.y - b.y);
    }
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", (e) => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved += Math.abs(dx) + Math.abs(dy);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const now = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) zoom(pinch / Math.max(1, now));
      pinch = now;
      const scale = view.distance / 600;
      pan(-dx * scale / 2, dy * scale / 2);
      return;
    }
    if (panning) {
      const scale = view.distance / 600;
      pan(-dx * scale, dy * scale);
    } else {
      view.azimuth -= dx * 0.006;
      view.polar = Math.max(0.08, Math.min(1.52, view.polar - dy * 0.006));
    }
  });
  function release(e) {
    const was = pointers.has(e.pointerId);
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = 0;
    if (was && e.type === "pointerup" && moved < 5 && e.target === canvas) pick(e);
  }
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom(1 + Math.sign(e.deltaY) * 0.1); }, { passive: false });
  canvas.addEventListener("dblclick", (e) => {
    // Fly to the spot on the ground that was double-clicked.
    const r = canvas.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) {
      freeLook();
      view.focus = new THREE.Vector3(hit.x, 1, hit.z);
    }
  });
  // Keyboard: WASD / arrows move, Q/E turn, +/- zoom, F follows Fuzzy again.
  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea, select")) return;
    const k = e.key.toLowerCase();
    if (k === "f") { follow.checked = true; view.focus = null; return; }
    if (["w", "a", "s", "d", "q", "e", "arrowup", "arrowdown", "arrowleft", "arrowright", "+", "=", "-"].includes(k)) {
      keys.add(k);
      if (k.startsWith("arrow")) e.preventDefault();
    }
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => keys.clear());
  function steer(dt) {
    if (!keys.size) return;
    const speed = Math.max(4, view.distance * 0.8) * dt;
    let right = 0, forward = 0;
    if (keys.has("w") || keys.has("arrowup")) forward += speed;
    if (keys.has("s") || keys.has("arrowdown")) forward -= speed;
    if (keys.has("d") || keys.has("arrowright")) right += speed;
    if (keys.has("a") || keys.has("arrowleft")) right -= speed;
    if (right || forward) pan(right, forward);
    if (keys.has("q")) view.azimuth += 1.5 * dt;
    if (keys.has("e")) view.azimuth -= 1.5 * dt;
    if (keys.has("+") || keys.has("=")) zoom(1 - 1.2 * dt);
    if (keys.has("-")) zoom(1 + 1.2 * dt);
  }

  const raycaster = new THREE.Raycaster();
  function pick(e) {
    const r = canvas.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hits = raycaster.intersectObjects([...objects.values()].map((o) => o.node), true);
    for (const hit of hits) {
      let n = hit.object;
      while (n && !n.userData.id) n = n.parent;
      if (n) { select(n.userData.id); return; }
    }
  }
  function select(id) {
    const entry = objects.get(id);
    const box = clear($("picked"));
    if (!entry) return;
    follow.checked = false;
    view.focus = new THREE.Vector3(entry.data.x, 1, entry.data.z);
    box.append(el("strong", {}, entry.data.name), el("div", {}, entry.data.description || ""),
      el("div", { class: "muted" }, "built " + ago(entry.data.created_at)));
  }

  // -------------------------------------------------------------- the panel
  function describeAction() {
    const verbs = { standing: "standing around", walking: "walking", building: "building", thinking: "thinking", resting: "resting", talking: "talking with someone", creating: "making something" };
    $("now").textContent = name + " is " + (verbs[body.action] || body.action);
    $("thought").textContent = body.thought || "";
    $("size").textContent = Math.round(world.size) + " m across";
    const bubble = $("bubble");
    bubble.textContent = body.thought || "";
    bubble.classList.toggle("hidden", !body.thought);
  }
  function listObjects() {
    const box = clear($("objects"));
    const items = [...objects.values()].map((o) => o.data).sort((a, b) => a.created_at < b.created_at ? 1 : -1);
    if (!items.length) { box.append(el("p", { class: "muted small" }, "Nothing yet. The ground is empty.")); return; }
    box.append(el("ul", { class: "plain" }, items.map((o) => el("li", {},
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); select(o.id); } }, o.name)))));
  }
  function describeAvatar() {
    $("avatar").textContent = avatarData && avatarData.description ? avatarData.description : name + " hasn’t designed how it looks yet.";
  }

  for (const o of world.objects) place(o, false);
  describeAction(); listObjects(); describeAvatar();

  live("/api/world/stream", (e) => {
    const p = e.payload || {};
    if (e.type === "world.body") { body = p.body; describeAction(); }
    else if (e.type === "world.size") { setSize(p.size); $("size").textContent = Math.round(p.size) + " m across"; }
    else if (e.type === "world.avatar") { avatarData = p.avatar; buildAvatar(); describeAvatar(); }
    else if (e.type === "world.object") {
      if (p.op === "removed") remove(p.object.id); else place(p.object, p.op === "built");
      listObjects();
    }
  });

  // ------------------------------------------------------------------ loop
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  const projected = new THREE.Vector3();
  // The thought bubble follows Fuzzy's head, but smoothly and always fully on screen, so turning
  // the view doesn't fling it about. It hides when Fuzzy is behind the camera or out of view.
  const bubblePos = { x: null, y: null };
  function placeBubble() {
    const bubble = $("bubble");
    if (bubble.classList.contains("hidden")) { bubblePos.x = null; return; }
    projected.set(shown.x, 2.3, shown.z).project(camera);
    const onScreen = projected.z < 1 && Math.abs(projected.x) <= 1.05 && Math.abs(projected.y) <= 1.05;
    bubble.style.visibility = onScreen ? "visible" : "hidden";
    if (!onScreen) { bubblePos.x = null; return; }
    const W = stage.clientWidth, H = stage.clientHeight, bw = bubble.offsetWidth, bh = bubble.offsetHeight, pad = 8;
    let x = ((projected.x + 1) / 2) * W, y = ((1 - projected.y) / 2) * H;
    x = Math.max(bw / 2 + pad, Math.min(W - bw / 2 - pad, x));
    y = Math.max(bh + pad, Math.min(H - pad, y));
    if (bubblePos.x === null) { bubblePos.x = x; bubblePos.y = y; }
    bubblePos.x += (x - bubblePos.x) * 0.3;
    bubblePos.y += (y - bubblePos.y) * 0.3;
    bubble.style.left = bubblePos.x.toFixed(1) + "px";
    bubble.style.top = bubblePos.y.toFixed(1) + "px";
  }
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    // Glide toward where the server says Fuzzy is.
    const k = 1 - Math.pow(0.001, dt);
    shown.x += (body.x - shown.x) * k; shown.z += (body.z - shown.z) * k;
    let dh = ((body.heading - shown.heading + 540) % 360) - 180;
    shown.heading += dh * k;
    const t = now / 1000, action = body.action;
    avatar.position.set(shown.x, 0, shown.z);
    avatar.rotation.set(0, rad(shown.heading), 0);
    let bob = 0, sway = 0, squash = 1;
    if (action === "walking") { bob = Math.abs(Math.sin(t * 8)) * 0.07; sway = Math.sin(t * 8) * 0.05; }
    else if (action === "building" || action === "creating") { bob = Math.abs(Math.sin(t * 5)) * 0.05; sway = Math.sin(t * 5) * 0.08; }
    else if (action === "thinking") { sway = Math.sin(t * 1.5) * 0.06; }
    else if (action === "talking") { bob = Math.abs(Math.sin(t * 4)) * 0.03; }
    else if (action === "resting") { squash = 0.92 + Math.sin(t * 1.2) * 0.01; }
    else { squash = 1 + Math.sin(t * 1.6) * 0.01; }
    avatar.position.y = bob; avatar.rotation.z = sway; avatar.scale.set(1, squash, 1);

    for (const entry of objects.values()) {
      if (entry.born !== null) {
        const f = Math.min(1, (now - entry.born) / 1200);
        const s = 1 - Math.pow(1 - f, 3);
        entry.node.scale.set(Math.max(0.01, s), Math.max(0.01, s), Math.max(0.01, s));
        if (f >= 1) entry.born = null;
      }
    }
    steer(dt);
    updateCamera();
    placeBubble();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
