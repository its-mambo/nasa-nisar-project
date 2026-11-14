"use strict";
// NISAR Change tab: T0 / T1 / Split compare with a draggable divider,
// independent detector overlay and HH grayscale controls, zoom/pan and full screen.

(() => {
  const el = id => document.getElementById(id);
  const change = el("change"), canvas = el("image"), message = el("message");
  const handle = el("handle"), split = el("split");
  const requested = new URLSearchParams(location.search).get("v");
  let revision = null;

  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const LABELS = {t0: "T0 — earlier image", t1: "T1 — later image", split: "Split view: T0 left, T1 right"};

  const state = {mode: "split", grayscale: false, showChange: true, split: 50, images: null};

  // ---------- Canvas drawing ----------
  const view = new RasterViewport(canvas, redraw, {wheelZoom: true, upscale: true});
  view.enableDrag(0);
  window.addEventListener("pagehide",() => {
    const viewport = Nisar.captureViewport(view);
    if (viewport) Nisar.view.save(revision,{viewport});
  });

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
    const t0 = state.grayscale ? imgs.t0_hh : imgs.t0;
    const t1 = state.grayscale ? imgs.t1_hh : imgs.t1;
    if (state.mode === "t0") drawLayer(ctx, t0, 0, view.width);
    else if (state.mode === "t1") drawLayer(ctx, t1, 0, view.width);
    else {
      // Split: the divider is fixed to the screen, convert it to a raster column.
      const col = (state.split / 100 * view.w - view.x) / view.scale;
      drawLayer(ctx, t0, 0, col);
      drawLayer(ctx, t1, col, view.width);
    }
    // The detector mask is on the same spatial grid for both acquisition dates.
    // Composite it after the base imagery so it stays blue in grayscale mode.
    if (state.showChange && imgs.overlay) drawLayer(ctx, imgs.overlay, 0, view.width);
  }

  // ---------- UI state ----------
  function setMode(mode) {
    state.mode = mode;
    change.dataset.mode = mode;
    document.querySelectorAll("[data-view]").forEach(b =>
      b.setAttribute("aria-pressed", String(b.dataset.view === mode)));
    updateLabel();
    view.invalidate();
    Nisar.view.save(revision, {viewerMode:mode, ...(["t0","t1"].includes(mode) ? {date:mode} : {})});
  }

  function setGrayscale(on) {
    state.grayscale = on;
    el("grayscale").setAttribute("aria-checked", String(on));
    updateLabel();
    view.invalidate();
    Nisar.view.save(revision, {grayscale:on});
  }

  function setChange(on) {
    state.showChange = on;
    el("overlay").setAttribute("aria-checked", String(on));
    el("changeLegend").parentElement.hidden = !on;
    updateLabel();
    view.invalidate();
    Nisar.view.save(revision, {showChange:on});
  }

  function setSplit(pct) {
    state.split = Math.max(0, Math.min(100, pct));
    const rounded = Math.round(state.split);
    change.style.setProperty("--split", state.split + "%");
    handle.setAttribute("aria-valuenow", String(rounded));
    handle.setAttribute("aria-valuetext", `${rounded}%`);
    view.invalidate();
    Nisar.view.save(revision, {split:state.split});
  }

  function updateLabel() {
    canvas.setAttribute("aria-label", `${LABELS[state.mode]} · ${state.grayscale ? "HH grayscale" : "HH/HV colour"} · ${state.showChange ? "change shown in blue" : "change hidden"}`);
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
  el("overlay").addEventListener("click", () => setChange(!state.showChange));
  el("grayscale").addEventListener("click", () => setGrayscale(!state.grayscale));

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
    setGrayscale(false);
    setChange(true);
    // Use the result's own acquisition dates, even if a new pair has been loaded.
    let info = null;
    try {
      const response = await fetch("/result" + (requested ? "?v=" + encodeURIComponent(requested) : ""), {cache: "no-store"});
      if (response.ok) info = await response.json();
    } catch { /* handled below */ }
    const acq = info?.result?.acquisitions || {};
    const d0 = formatDate(acq.t0?.zeroDopplerStartTime), d1 = formatDate(acq.t1?.zeroDopplerStartTime);
    if (d0) el("pillBeforeText").textContent = `Before · T0 · ${d0}`;
    if (d1) el("pillAfterText").textContent = `After · T1 · ${d1}`;

    revision = Nisar.isResultRevision(requested) ? requested
             : info?.result_revision != null ? String(info.result_revision) : null;
    if (!revision) {
      showMessage("Run an analysis on the Pick an area tab first.", {href: "/", text: "Pick an area"});
      return;
    }
    document.querySelector('.app-tabs a[href$="#results"]').href = `/?v=${encodeURIComponent(revision)}#results`;
    if (info?.result?.sample) {
      el("viewerNotice").textContent = "Saved analysis sample · fixed files, ROI, and threshold 0.10. Display controls do not rerun the model.";
      el("viewerNotice").hidden = false;
    }
    try {
      const [t0, t1] = await Promise.all(["t0", "t1"].map(loadImage));
      state.images = {t0, t1};
    } catch {
      showMessage("This result is no longer available.", {href: "/change", text: "Open the latest result"});
      return;
    }
    const extras = await Promise.allSettled(["t0_hh", "t1_hh", "overlay"].map(async name => {
      state.images[name] = await loadImage(name);
    }));
    el("grayscale").disabled = !(state.images.t0_hh && state.images.t1_hh);
    el("overlay").disabled = !state.images.overlay;
    const saved = Nisar.view.read(revision);
    setGrayscale(!el("grayscale").disabled && (saved.grayscale ?? false));
    setChange(!el("overlay").disabled && (saved.showChange ?? true));
    setMode(saved.viewerMode || saved.date || "split");
    setSplit(saved.split ?? 50);
    message.hidden = true;
    view.setRaster(state.images.t0.naturalWidth, state.images.t0.naturalHeight);
    Nisar.restoreViewport(view,saved.viewport);
    if (extras.some(item => item.status === "rejected")) {
      el("viewerNotice").textContent = "Some display layers are unavailable for this result. Run a new analysis to generate them.";
      el("viewerNotice").hidden = false;
    }
  }

  init();
})();
