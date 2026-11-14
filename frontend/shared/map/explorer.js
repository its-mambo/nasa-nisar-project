"use strict";

// Shared by Results and the Change viewer; preferences belong to a result revision.
(() => {
  function createPreferences(storage) {
    const memory = new Map();
    function clean(value) {
      const result = {};
      if (!value || typeof value !== "object") return result;
      for (const key of ["grayscale", "showChange"]) if (typeof value[key] === "boolean") result[key] = value[key];
      if (["t0","t1"].includes(value.date)) result.date = value.date;
      if (["t0","t1","split"].includes(value.viewerMode)) result.viewerMode = value.viewerMode;
      if (Number.isFinite(value.split)) result.split = Math.max(0,Math.min(100,value.split));
      const v = value.viewport;
      if (v && [v.zoom,v.x,v.y].every(Number.isFinite) && v.zoom > 0) {
        result.viewport = {zoom:Math.max(1,Math.min(8,v.zoom)),x:Math.max(0,Math.min(1,v.x)),y:Math.max(0,Math.min(1,v.y))};
      }
      return result;
    }
    const key = revision => `nisar-view-v1:${revision}`;
    function read(revision) {
      if (revision == null) return {};
      try {
        const raw = storage()?.getItem(key(revision));
        if (raw) return clean(JSON.parse(raw));
      } catch { /* Memory fallback also works when browser storage is unavailable. */ }
      return {...memory.get(key(revision))};
    }
    return {read, save(revision, patch) {
      if (revision == null) return;
      const value = clean({...read(revision), ...patch});
      memory.set(key(revision), value);
      try { storage()?.setItem(key(revision),JSON.stringify(value)); } catch { /* use memory */ }
    }};
  }

  function crsName(code) {
    const n = Number(code);
    if (n >= 32601 && n <= 32660) return `WGS 84 / UTM zone ${n-32600}N`;
    if (n >= 32701 && n <= 32760) return `WGS 84 / UTM zone ${n-32700}S`;
    return ({3413:"WGS 84 / NSIDC Sea Ice Polar Stereographic North",3031:"WGS 84 / Antarctic Polar Stereographic"})[n] || "Projected map coordinate system";
  }
  function captureViewport(view) {
    if (!view.width || !view.height || !view.w || !view.h) return null;
    const [x,y] = view.pixel(view.w/2,view.h/2);
    return {zoom:view.scale/view.fitScale,x:x/view.width,y:y/view.height};
  }
  function restoreViewport(view, saved) {
    if (!saved || !view.width || !view.height) return;
    view.scale = Math.max(view.fitScale,Math.min(view.maxScale(),view.fitScale*saved.zoom));
    view.x = view.w/2-saved.x*view.width*view.scale;
    view.y = view.h/2-saved.y*view.height*view.scale;
    view.clamp(); view.invalidate();
  }
  function openProjection(code) {
    if (code == null || !Number.isInteger(Number(code))) return;
    let dialog = document.getElementById("projectionDialog");
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.id = "projectionDialog"; dialog.className = "info-dialog";
      dialog.setAttribute("aria-labelledby","projectionTitle");
      const head = document.createElement("div"); head.className = "info-dialog-head";
      const title = document.createElement("h2"); title.id = "projectionTitle"; title.textContent = "What does EPSG mean?";
      const close = document.createElement("button"); close.className = "dialog-close"; close.textContent = "×";
      close.setAttribute("aria-label","Close map reference explanation"); close.type = "button";
      close.addEventListener("click", () => dialog.close());
      head.append(title,close);
      const body = document.createElement("div"); body.className = "info-dialog-body";
      for (const text of [
        "An EPSG code is an identifier for a coordinate reference system: the rules that place image pixels on a map of Earth.",
        "These GCOV images use a map grid measured in metres. The app uses that grid to align the two dates and calculate projected areas.",
        "When working with the measurements in GIS software, use the same EPSG code to keep the coordinates aligned."
      ]) { const p = document.createElement("p"); p.textContent = text; body.append(p); }
      const name = document.createElement("p"); name.id = "projectionName"; name.className = "projection-name"; body.prepend(name);
      const source = document.createElement("a"); source.href = "https://nisar-docs.asf.alaska.edu/projections/";
      source.textContent = "NISAR map projections guide ↗"; source.target = "_blank"; source.rel = "noopener";
      body.append(source); dialog.append(head,body); document.body.append(dialog);
      dialog.addEventListener("click",e => { if (e.target === dialog) dialog.close(); });
    }
    document.getElementById("projectionName").textContent = `EPSG:${code} · ${crsName(code)}`;
    dialog.showModal();
  }
  const api = {view:createPreferences(() => globalThis.sessionStorage), crsName, openProjection, createPreferences,captureViewport,restoreViewport};
  if (typeof window !== "undefined") window.Nisar = api;
  if (typeof module !== "undefined") module.exports = api;
})();
