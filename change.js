"use strict";
// NISAR Change tab: T0 / T1 / Split compare with a draggable divider,
// colour ↔ grayscale toggle, zoom/pan and full screen.

(() => {
  const el = id => document.getElementById(id);
  const change = el("change"), canvas = el("image"), message = el("message");
  const handle = el("handle"), split = el("split");
  const requested = new URLSearchParams(location.search).get("v");
  let revision = null;

  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const LABELS = {t0: "T0 — earlier image", t1: "T1 — later image", split: "Split view: T0 left, T1 right"};

  const state = {mode: "split", colour: true, split: 50, images: null};

  // ---------- Canvas drawing ----------
  const view = new RasterViewport(canvas, redraw, {wheelZoom: true, upscale: true});
  view.enableDrag(0);

  // Draws `img` into raster coordinates, limited to raster columns [c0, c1).
  function drawLayer(ctx, img, c0, c1) {
    const [x, y, X, Y] = view.visible();
    const left = Math.max(x, c0), right = Math.min(X, c1);
    if (right <= left || Y <= y) return;
    // T1 may differ in size from T0; map it onto the T0 raster grid.
    const sx = img.naturalWidth / view.width, sy = img.naturalHeight / view.height;
    ctx.drawImage(img, left * sx, y * sy, (right - left) * sx, (Y - y) * sy,
                  left, y, right - left, Y - y);
  }

  function redraw(ctx) {
    const imgs = state.images;
    if (!imgs) return;
    if (state.mode === "t0") return drawLayer(ctx, imgs.t0, 0, view.width);
    if (state.mode === "t1") return drawLayer(ctx, imgs.t1, 0, view.width);
    // Split: the divider is fixed to the screen, convert it to a raster column.
    const col = (state.split / 100 * view.w - view.x) / view.scale;
    drawLayer(ctx, imgs.t0, 0, col);
    drawLayer(ctx, imgs.t1, col, view.width);
  }

  // ---------- UI state ----------
  function setMode(mode) {
    state.mode = mode;
    change.dataset.mode = mode;
    document.querySelectorAll("[data-view]").forEach(b =>
      b.setAttribute("aria-pressed", String(b.dataset.view === mode)));
    updateLabel();
    view.invalidate();
  }

  // Colour ↔ grayscale only (CSS filter on the canvas); the images stay the same.
  function setColour(on) {
    state.colour = on;
    el("overlay").setAttribute("aria-checked", String(on));
    change.classList.toggle("grayscale", !on);
    updateLabel();
  }

  function setSplit(pct) {
    state.split = Math.max(0, Math.min(100, pct));
    const rounded = Math.round(state.split);
    change.style.setProperty("--split", state.split + "%");
    handle.setAttribute("aria-valuenow", String(rounded));
    handle.setAttribute("aria-valuetext", `${rounded}%`);
    view.invalidate();
  }

  function updateLabel() {
    canvas.setAttribute("aria-label", `${LABELS[state.mode]} · ${state.colour ? "colour" : "grayscale"}`);
  }

  function showMessage(text, link) {
    message.replaceChildren(text);
    if (link) {
      const a = document.createElement("a");
      a.href = link.href;
      a.textContent = link.text;
      message.append(" ", a);
    }
    message.hidden = false;
  }

  document.querySelectorAll("[data-view]").forEach(b => b.addEventListener("click", () => setMode(b.dataset.view)));
  el("overlay").addEventListener("click", () => setColour(!state.colour));

  // ---------- Divider drag (pointer / touch / mouse) ----------
  let dragId = null;
  function pctFromEvent(e) {
    const r = change.getBoundingClientRect();
    return (e.clientX - r.left) / r.width * 100;
  }
  for (const target of [handle, split]) {
    target.addEventListener("pointerdown", e => {
      if (!e.isPrimary || e.button !== 0) return;
      e.preventDefault();
      dragId = e.pointerId;
      target.setPointerCapture(e.pointerId);
      if (target === handle) handle.focus({preventScroll: true});
      setSplit(pctFromEvent(e));
    });
    target.addEventListener("pointermove", e => { if (e.pointerId === dragId) setSplit(pctFromEvent(e)); });
    target.addEventListener("lostpointercapture", e => { if (e.pointerId === dragId) dragId = null; });
  }
  handle.addEventListener("keydown", e => {
    const step = {ArrowLeft: -2, ArrowRight: 2, ArrowDown: -2, ArrowUp: 2}[e.key];
    if (step !== undefined) { e.preventDefault(); e.stopPropagation(); setSplit(state.split + step); }
    else if (e.key === "Home") { e.preventDefault(); setSplit(0); }
    else if (e.key === "End") { e.preventDefault(); setSplit(100); }
  });

  // ---------- Zoom ----------
  const STEP = 1.25;
  const zoomIn = () => view.zoomAt(STEP, view.w / 2, view.h / 2);
  const zoomOut = () => view.zoomAt(1 / STEP, view.w / 2, view.h / 2);
  el("zoomIn").addEventListener("click", zoomIn);
  el("zoomOut").addEventListener("click", zoomOut);
  el("recenter").addEventListener("click", () => view.reset());

  // ---------- Full screen ----------
  const fsApi = !!(change.requestFullscreen || change.webkitRequestFullscreen) &&
                (document.fullscreenEnabled ?? document.webkitFullscreenEnabled ?? true);
  const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement || null;
  let fallbackFs = false;

  function applyFsUi(on) {
    change.classList.toggle("is-fs", on);
    change.classList.toggle("fs-fallback", on && fallbackFs);
    el("fullscreenText").textContent = on ? "Exit full screen" : "Full screen";
    el("fullscreenIcon").src = on ? "/icons/exit-fullscreen.svg" : "/icons/fullscreen.svg";
    el("fullscreen").setAttribute("aria-label", on ? "Exit full screen (Esc)" : "Full screen (F)");
    // Canvas size changes: RasterViewport's ResizeObserver refits, but redraw now as well.
    view.invalidate();
  }
  const isFs = () => change.classList.contains("is-fs");

  async function enterFs() {
    if (fsApi) {
      try {
        await (change.requestFullscreen ? change.requestFullscreen() : change.webkitRequestFullscreen());
        return;
      } catch { /* fall back to CSS below */ }
    }
    fallbackFs = true;
    applyFsUi(true);
  }
  function exitFs() {
    if (fallbackFs) { fallbackFs = false; applyFsUi(false); return; }
    if (fsElement()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  }
  function onFsChange() { if (!fallbackFs) applyFsUi(fsElement() === change); }
  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);
  el("fullscreen").addEventListener("click", () => (isFs() ? exitFs() : enterFs()));

  // ---------- Keyboard shortcuts ----------
  document.addEventListener("keydown", e => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "Escape" && fallbackFs) { e.preventDefault(); exitFs(); return; }
    const actions = {
      "1": () => setMode("t0"), "2": () => setMode("t1"), "3": () => setMode("split"),
      "f": () => (isFs() ? exitFs() : enterFs()), "F": () => (isFs() ? exitFs() : enterFs()),
      "+": zoomIn, "=": zoomIn, "-": zoomOut, "_": zoomOut, "0": () => view.reset(),
    };
    const run = actions[e.key];
    if (!run) return;
    e.preventDefault();
    run();
  });

  // ---------- Data ----------
  function formatDate(value) {
    if (!value) return null;
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return null;
    return `${String(d.getUTCDate()).padStart(2, "0")} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  }

  async function loadImage(name) {
    const image = new Image();
    image.src = `/${name}.png?v=${revision}`;
    await image.decode();
    return image;
  }

  async function init() {
    setSplit(50);
    setMode("split");
    setColour(true);
    // /pair gives the acquisition dates and, without ?v=, the latest result revision.
    let info = null;
    try {
      const response = await fetch("/pair", {cache: "no-store"});
      if (response.ok) info = await response.json();
    } catch { /* handled below */ }
    const acq = info?.pair?.acquisitions || {};
    const d0 = formatDate(acq.t0?.zeroDopplerStartTime), d1 = formatDate(acq.t1?.zeroDopplerStartTime);
    if (d0) el("pillBeforeText").textContent = `Before · T0 · ${d0}`;
    if (d1) el("pillAfterText").textContent = `After · T1 · ${d1}`;

    revision = /^\d+$/.test(requested || "") ? requested
             : info?.result_revision != null ? String(info.result_revision) : null;
    if (!revision) {
      showMessage("Run an analysis on the Pick tab first.", {href: "/", text: "Go to Pick"});
      return;
    }
    try {
      const [t0, t1] = await Promise.all(["t0", "t1"].map(loadImage));
      state.images = {t0, t1};
    } catch {
      showMessage("This result is no longer available.", {href: "/change", text: "Open the latest result"});
      return;
    }
    message.hidden = true;
    view.setRaster(state.images.t0.naturalWidth, state.images.t0.naturalHeight);
  }

  init();
})();
