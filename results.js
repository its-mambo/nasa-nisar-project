"use strict";

const resultsEl = id => document.getElementById(id);
const resultCanvas = resultsEl("resultMap");
let resultImage = null;
let resultOverlay = null;
let resultMode = "t0";
let resultShowChange = true;
let resultGrayscale = false;
let resultLayers = new Map();
let resultLoadId = 0;
let resultData = null;
const resultViewport = new RasterViewport(resultCanvas, ctx => {
  if (!resultImage) return;
  const [x, y, right, bottom] = resultViewport.visible();
  if (right > x && bottom > y) {
    ctx.drawImage(resultImage, x, y, right - x, bottom - y, x, y, right - x, bottom - y);
    if (resultShowChange && resultOverlay) {
      ctx.drawImage(resultOverlay, x, y, right - x, bottom - y, x, y, right - x, bottom - y);
    }
  }
});
resultViewport.enableDrag(0);

const resultNumber = value => Number(value || 0).toLocaleString("en-US");
const resultArea = value => Number(value || 0).toLocaleString("en-US", {maximumFractionDigits: 2, minimumFractionDigits: 2});
function acquisitionDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}
function setResultText(id, value) { resultsEl(id).textContent = value; }

async function showResultMode(mode) {
  if (!resultData) return;
  resultMode = mode;
  document.querySelectorAll("[data-result-mode]").forEach(button =>
    button.setAttribute("aria-pressed", String(button.dataset.resultMode === mode)));
  const grayscale = resultGrayscale;
  const file = `${mode}${grayscale ? "_hh" : ""}.png`;
  const label = `${mode === "t0" ? "T0 · Before" : "T1 · After"} · ${grayscale ? "HH grayscale" : "HH/HV colour"} · ${resultShowChange ? "change shown in blue" : "change hidden"}`;
  const id = ++resultLoadId;
  function load(file) {
    const url = resultData.images?.[file];
    if (!url) return Promise.reject(new Error("Image unavailable for this result. Run a new analysis to generate its display layers."));
    if (!resultLayers.has(url)) {
      const img = new Image();
      img.src = url;
      const cache = resultLayers;
      cache.set(url,img.decode().then(() => img).catch(err => { cache.delete(url); throw err; }));
    }
    return resultLayers.get(url);
  }
  const showMask = resultShowChange;
  resultsEl("resultShowChange").setAttribute("aria-checked", String(showMask));
  resultsEl("resultGrayscale").setAttribute("aria-checked", String(grayscale));
  resultsEl("resultMaskLegend").hidden = !showMask;
  setResultText("resultMapLabel", `${mode.toUpperCase()} · ${grayscale ? "HH GRAYSCALE" : "COLOUR"} · ${showMask ? "CHANGE SHOWN" : "CHANGE HIDDEN"}`);
  setResultText("resultMapStatus", "Loading " + label + "…");
  resultsEl("resultMapStatus").hidden = false;
  resultsEl("resultDownload").hidden = true;
  try {
    const [img, overlay] = await Promise.all([load(file), showMask ? load("overlay.png") : null]);
    if (id !== resultLoadId) return;
    resultImage = img; resultOverlay = overlay;
    resultCanvas.setAttribute("aria-label", label);
    if (resultViewport.width !== img.naturalWidth || resultViewport.height !== img.naturalHeight) {
      resultViewport.setRaster(img.naturalWidth, img.naturalHeight);
    } else resultViewport.invalidate();
    resultsEl("resultMapStatus").hidden = true;
    const changeName = mode === "t0" ? "change" : "change_t1";
    const download = showMask ? `${changeName}${grayscale ? "_hh" : ""}.png` : file;
    const downloadUrl = resultData.images?.[download];
    resultsEl("resultDownload").hidden = !downloadUrl;
    if (downloadUrl) {
      resultsEl("resultDownload").href = downloadUrl;
      resultsEl("resultDownload").download = download;
    }
  } catch (err) {
    if (id === resultLoadId) setResultText("resultMapStatus", err.message || "Could not load this result image.");
  }
}

function resultZoom(factor) {
  if (!resultViewport.width) return;
  resultViewport.scale = Math.max(resultViewport.fitScale, Math.min(8, resultViewport.scale * factor));
  resultViewport.x = (resultViewport.w - resultViewport.width * resultViewport.scale) / 2;
  resultViewport.y = (resultViewport.h - resultViewport.height * resultViewport.scale) / 2;
  resultViewport.clamp();
  resultViewport.invalidate();
}

function renderResultsV2(data) {
  const newResult = resultData !== data;
  resultData = data;
  if (newResult) {
    resultLayers = new Map();
    resultImage = resultOverlay = null;
    resultViewport.setRaster(0,0);
    resultMode = "t0";
    resultShowChange = !!data.images?.["overlay.png"];
    resultGrayscale = false;
  }
  resultsEl("resultShowChange").disabled = !data.images?.["overlay.png"];
  resultsEl("resultGrayscale").disabled = !(data.images?.["t0_hh.png"] && data.images?.["t1_hh.png"]);
  const roi = data.roi || {};
  const acq = data.acquisitions || {};
  const start = acquisitionDate(acq.t0?.zeroDopplerStartTime);
  const end = acquisitionDate(acq.t1?.zeroDopplerStartTime);
  const dateFormat = new Intl.DateTimeFormat("en-GB", {day:"2-digit",month:"short",year:"numeric",timeZone:"UTC"});
  setResultText("resultDates", start && end ? `${dateFormat.format(start)} → ${dateFormat.format(end)}` : "Acquisition dates unavailable");
  const days = start && end ? Math.round((Date.UTC(end.getUTCFullYear(),end.getUTCMonth(),end.getUTCDate()) - Date.UTC(start.getUTCFullYear(),start.getUTCMonth(),start.getUTCDate())) / 86400000) : null;
  setResultText("resultDays", days === null ? "Date interval unavailable" : `${days} days`);
  setResultText("resultEpsg", roi.epsg ? `EPSG:${roi.epsg}` : "Projection unavailable");
  setResultText("resultMapScale", `${roi.model_spacing_m ?? "—"} m grid · EPSG:${roi.epsg ?? "—"}`);
  setResultText("resultFlaggedArea", `${resultArea(data.changed_pixels * (roi.model_spacing_m || 0) ** 2 / 1e6)} km²`);
  const pct = data.valid_pixels ? 100 * data.changed_pixels / data.valid_pixels : 0;
  setResultText("resultFlaggedPct", `${pct.toFixed(1)}% of valid model pixels · threshold ${data.threshold ?? "—"}`);
  setResultText("resultFlaggedCount", resultNumber(data.changed_pixels));
  setResultText("resultModelPct", `${pct.toFixed(1)}% flagged`);
  setResultText("resultUnflaggedPct", `${(100-pct).toFixed(1)}% not flagged`);
  resultsEl("resultSplitFill").style.width = `${Math.max(0,Math.min(100,pct))}%`;
  const prediction = data.predictions?.linear;
  const score = prediction?.scores?.[prediction.label];
  setResultText("resultModelScore", Number.isFinite(score) ? `${(score*100).toFixed(1)}%` : "—");
  setResultText("resultModelLabel", prediction ? `${({water_related:"Water-related change",other_change:"Other change"}[prediction.label] || prediction.label)} score` : "No classifier label");
  setResultText("resultCoverage", `${resultArea(roi.projected_rectangle_km2)} km² · EPSG:${roi.epsg ?? "—"}`);
  const sectors = [...(data.spatial_context?.zones || [])].filter(z => z.flagged).sort((a,b) => b.flagged-a.flagged).slice(0,3);
  setResultText("resultSectors", sectors.length ? sectors.map(z => z.zone.replace(/\b\w/g, c => c.toUpperCase())).join(" · ") : "No flagged sectors");
  const geo = acq.t0 || {};
  setResultText("resultGeometry", [geo.orbitPassDirection, geo.lookDirection, geo.trackNumber == null ? null : `Track ${geo.trackNumber}`, geo.frameNumber == null ? null : `Frame ${geo.frameNumber}`].filter(Boolean).join(" · ") || "Acquisition geometry unavailable");
  const pairs = (data.paired_sector_deltas || []).filter(row => row.flagged?.median_delta_hh_hv_db && row.unflagged?.median_delta_hh_hv_db);
  if (pairs.length) {
    const text = pairs.map(row => {
      const flagged = row.flagged.median_delta_hh_hv_db;
      const background = row.unflagged.median_delta_hh_hv_db;
      return `${row.zone}: median HH ${flagged[0] >= 0 ? "+" : ""}${flagged[0].toFixed(2)} dB, HV ${flagged[1] >= 0 ? "+" : ""}${flagged[1].toFixed(2)} dB in flagged cells; HH ${background[0] >= 0 ? "+" : ""}${background[0].toFixed(2)} dB, HV ${background[1] >= 0 ? "+" : ""}${background[1].toFixed(2)} dB in unflagged cells.`;
    }).join("\n");
    setResultText("resultPattern", text);
  } else {
    setResultText("resultPattern", data.spatial_context?.change_map_description || "No paired sector comparison is available for this selection.");
  }
  setResultText("resultPatternNote", data.interpretation?.source === "visual" ? "AI interpretation is unvalidated. Blue is not a water map." : "Measured SAR values describe change, not its physical cause.");
  resultsEl("resultViewerLink").disabled = resultRevision() === null;
  showResultMode(resultMode);
}

// Result revision: the ?v= of the result image URLs (same source as /t0.png?v=… above).
function resultRevision() {
  const url = resultData?.images?.["t0.png"];
  if (!url) return null;
  const v = new URL(url, location.origin).searchParams.get("v");
  return /^\d+$/.test(v || "") ? v : null;
}
resultsEl("resultViewerLink").addEventListener("click", () => {
  const rev = resultRevision();
  if (rev === null) return;
  location.href = `/change?v=${rev}`;
});

document.querySelectorAll("[data-result-mode]").forEach(button => button.addEventListener("click", () => showResultMode(button.dataset.resultMode)));
resultsEl("resultShowChange").addEventListener("click", () => {
  resultShowChange = !resultShowChange;
  showResultMode(resultMode);
});
resultsEl("resultGrayscale").addEventListener("click", () => {
  resultGrayscale = !resultGrayscale;
  showResultMode(resultMode);
});
resultsEl("resultZoomIn").addEventListener("click", () => resultZoom(1.5));
resultsEl("resultZoomOut").addEventListener("click", () => resultZoom(1/1.5));
resultsEl("resultOpenReport").addEventListener("click", () => resultsEl("resultReportDialog").showModal());
resultsEl("resultCloseReport").addEventListener("click", () => resultsEl("resultReportDialog").close());
resultsEl("resultReportDialog").addEventListener("click", e => { if (e.target === e.currentTarget) e.currentTarget.close(); });

function showInterpretation(data) {
  const root = resultsEl("hypotheses");
  root.replaceChildren();
  if (data.source === "visual") {
    const p = document.createElement("p");
    p.style.whiteSpace = "pre-wrap";
    p.textContent = data.summary === "" ? "Qwen returned no text." : data.summary;
    root.append(p);
    resultsEl("interpretationNote").textContent = (data.decoder === "qwen_lora" ? "Qwen + decoder LoRA. " : "Base Qwen. ") + "AI interpretation. The standalone map shows detector flags; the Results map overlays them in blue. " + data.limitation;
    return;
  }
  [data.primary,...data.alternatives].forEach((item,i) => {
    const heading = document.createElement("h4");
    heading.textContent = (i ? "Also worth checking: " : "") + item.title;
    root.append(heading);
    for (const [label,value] of [["What the radar shows",item.evidence],
                                ["Interpretation",item.possibility],
                                ["How to check",item.check]]) {
      const p = document.createElement("p");
      const strong = document.createElement("strong");
      strong.textContent = label + ": ";
      p.append(strong,value);
      root.append(p);
    }
  });
  resultsEl("interpretationNote").textContent = (data.visual_note ? data.visual_note + " " : "") + data.limitation +
    (data.source === "qwen" ? " Hypotheses ranked by Qwen." : " No decoder ranking was used.");
}

function renderDecoderInputs(root,data) {
  function element(tag,value) {
    const el = document.createElement(tag);
    if (value !== undefined) el.textContent = String(value);
    return el;
  }
  root.replaceChildren(element("p",
    "Qwen receives clean T0, clean T1, and a standalone detector map on the same grid." +
    (data.streams.length ? "" : " The run stopped before visual tokens were captured.")));
  const groups = new Map();
  for (const panel of data.panels) {
    if (!groups.has(panel.group)) groups.set(panel.group,[]);
    groups.get(panel.group).push(panel);
  }
  for (const [group,panels] of groups) {
    const w = data.windows[group-1];
    root.append(element("h4",w ? "Close-up " + w.number + " · " + w.zone + " · model-grid box [" + w.model_box.join(", ") + "]" : "Selected area overview"));
    const grid = element("div");
    grid.className = "decoder-panels";
    if (!group && data.windows.length) {
      const label = element("label"), toggle = element("input");
      toggle.type = "checkbox"; toggle.checked = true;
      toggle.addEventListener("change",() => grid.classList.toggle("hide-windows",!toggle.checked));
      label.append(toggle," Mark close-ups on the overview");
      root.append(label);
    }
    for (const panel of panels) {
      const figure = element("figure"), raster = element("a"), img = element("img");
      raster.className = "decoder-raster";
      raster.href = panel.url; raster.target = "_blank"; raster.rel = "noopener";
      img.src = panel.url; img.alt = panel.label; img.loading = "lazy";
      [img.width,img.height] = panel.pixels;
      raster.append(img);
      if (panel.slot === 0) {
        for (const close of data.windows) {
          const box = element("span"), n = element("span",close.number);
          const [x,y,w,h] = close.panel_rect.map((v,i) => v/panel.pixels[i%2]*100+"%");
          box.className = "decoder-window";
          box.style.left = x; box.style.top = y; box.style.width = w; box.style.height = h;
          box.setAttribute("aria-hidden","true");
          box.append(n); raster.append(box);
        }
      }
    const source = "Qwen vision";
      const [first,end] = panel.token_rows;
      figure.append(raster,element("figcaption",panel.label + " · " + source + " · " + panel.pixels.join(" × ") + " px · " + (end-first) + " visual token slots"));
      grid.append(figure);
    }
    root.append(grid);
  }
  if (data.streams.length) {
    root.append(element("p","Qwen visual streams: " +
      data.streams.map(s => s.name + " " + s.shape.join(" × ") + " (" + s.dtype + ")").join("; ") + "."));
  }
  if (data.tensor_url) {
    const a = element("a","Download visual token tensors (.npz)");
    a.href = data.tensor_url; a.download = "decoder-tokens.npz";
    root.append(a);
  }
  data.attempts.forEach((attempt,i) => {
    const d = element("details");
    d.append(element("summary","Attempt " + (i+1) + " · prompt (" + attempt.input_token_count + " input tokens) and raw response"));
    d.append(element("pre",attempt.prompt));
    d.append(element("p",attempt.reused_t0_t1_features ? "Reused cached T0/T1 features and encoded the standalone map separately." : "Encoded all three panels together."));
    d.append(element("h4","Raw response"));
    d.append(element("pre",attempt.raw_answer ?? "Generation did not produce a response."));
    root.append(d);
  });
}

const inspector = resultsEl("decoderInputs");
let loadedDecoderUrl;
inspector.addEventListener("toggle",async () => {
  const url = inspector.dataset.url;
  if (!inspector.open || !url || loadedDecoderUrl === url) return;
  loadedDecoderUrl = url;
  const root = resultsEl("decoderBody");
  root.textContent = "Loading decoder inputs…";
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error("Decoder inputs are no longer available for this result.");
    const data = await r.json();
    if (inspector.dataset.url === url) renderDecoderInputs(root,data);
  } catch (err) {
    if (inspector.dataset.url === url) {
      loadedDecoderUrl = undefined;
      root.textContent = err.message;
    }
  }
});


function showResultsV2(data) {
  resultsEl("analysisPage").hidden = true;
  resultsEl("pickTab").removeAttribute("aria-current");
  resultsEl("resultsTab").setAttribute("aria-current", "page");
  resultsEl("results").hidden = false;
  renderResultsV2(data);
  inspector.open = false;
  inspector.dataset.url = data.decoder_inputs_url || "";
  inspector.hidden = !data.decoder_inputs_url;
  loadedDecoderUrl = undefined;
  resultsEl("decoderBody").replaceChildren();
  const sections = resultsEl("reportSections");
  sections.replaceChildren();
  for (const item of data.report_sections || []) {
    const h = document.createElement("h4"), p = document.createElement("p");
    h.textContent = item.title;
    p.textContent = item.text;
    sections.append(h, p);
  }
  showInterpretation(data.interpretation);
  resultsEl("details").textContent = JSON.stringify(data,null,2);
  resultsEl("pickPage").hidden = true;
  // "/#results" reopens this tab (used by the Results link on the Change page).
  if (location.hash !== "#results") history.replaceState(null, "", "/#results");
  window.scrollTo(0,0);
}
function showPickTab() {
  if (resultsEl("modelProgress").getAttribute("aria-busy") === "true") {
    showAnalysisPage();
    return;
  }
  resultsEl("analysisPage").hidden = true;
  resultsEl("resultsTab").removeAttribute("aria-current");
  resultsEl("pickTab").setAttribute("aria-current", "page");
  resultsEl("results").hidden = true;
  resultsEl("pickPage").hidden = false;
  if (location.hash) history.replaceState(null, "", "/");
  window.scrollTo(0,0);
}
resultsEl("pickTab").addEventListener("click", e => {
  e.preventDefault();
  showPickTab();
});

// Open the Results tab from the URL: the latest result is kept by the server.
async function openResultsFromHash() {
  if (location.hash !== "#results") return;
  if (savedRun()) return;
  try {
    const r = await fetch("/result", {cache: "no-store"});
    const {result} = r.ok ? await r.json() : {};
    if (location.hash !== "#results") return;
    if (result) {
      showResultsV2(result);
    } else {
      showPickTab();
      status("No analysis result yet. Draw a selection and run the model.");
    }
  } catch {
    showPickTab();
  }
}
window.addEventListener("hashchange", openResultsFromHash);
window.addEventListener("DOMContentLoaded", openResultsFromHash);
