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
let investigationFocus = null, investigationFocusPending = false;
const resultViewport = new RasterViewport(resultCanvas, ctx => {
  if (!resultImage) return;
  const [x, y, right, bottom] = resultViewport.visible();
  if (right > x && bottom > y) {
    ctx.drawImage(resultImage, x, y, right - x, bottom - y, x, y, right - x, bottom - y);
    if (resultShowChange && resultOverlay) {
      ctx.drawImage(resultOverlay, x, y, right - x, bottom - y, x, y, right - x, bottom - y);
    }
    if (investigationFocus) {
      const box = NisarInvestigation.sectorBox(investigationFocus.zone);
      if (box) {
        const [a,b,c,d] = box.map((value, i) => Math.floor(value * (i % 2 ? resultViewport.height : resultViewport.width)));
        const unit = 1 / resultViewport.scale;
        ctx.save(); ctx.strokeStyle = "#ffc81e"; ctx.lineWidth = 2 * unit;
        ctx.setLineDash([8 * unit, 5 * unit]); ctx.strokeRect(a,b,c-a,d-b); ctx.setLineDash([]);
        const label = `Stop ${investigationFocus.number} · ${investigationFocus.zone} sector`;
        ctx.font = `600 ${12 * unit}px system-ui, sans-serif`;
        const labelWidth = ctx.measureText(label).width + 16 * unit;
        const labelX = Math.max(a, Math.min(c - labelWidth, right - labelWidth - 8 * unit));
        ctx.fillStyle = "#ffc81e"; ctx.fillRect(labelX,d - 24 * unit,labelWidth,24 * unit);
        ctx.fillStyle = "#14212b"; ctx.fillText(label,labelX + 8 * unit,d - 8 * unit);
        ctx.restore();
      }
    }
  }
});
resultViewport.enableDrag(0);
function focusInvestigationStop() {
  if (!investigationFocus || !resultViewport.width || !resultViewport.w) return;
  const box = NisarInvestigation.sectorBox(investigationFocus.zone);
  if (!box) return;
  const [x,y,right,bottom] = box;
  resultViewport.scale = Math.max(resultViewport.fitScale, Math.min(resultViewport.maxScale(),
    resultViewport.w / ((right-x) * resultViewport.width) * .82,
    resultViewport.h / ((bottom-y) * resultViewport.height) * .7));
  resultViewport.x = resultViewport.w / 2 - (x+right) / 2 * resultViewport.width * resultViewport.scale;
  resultViewport.y = resultViewport.h / 2 - (y+bottom) / 2 * resultViewport.height * resultViewport.scale;
  resultViewport.clamp(); resultViewport.invalidate(); investigationFocusPending = false;
  saveResultPreferences({viewport:Nisar.captureViewport(resultViewport)});
}
const investigation = NisarInvestigation.create({
  onSelect(stop) {
    investigationFocus = stop; investigationFocusPending = true;
    // Allow the newly visible guide panel to establish the map dimensions first.
    requestAnimationFrame(() => {
      if (investigationFocus !== stop) return;
      resultViewport.size(); focusInvestigationStop();
      resultCanvas.scrollIntoView({behavior:"smooth",block:"start"});
    });
  },
  onOverview() {
    investigationFocus = null; investigationFocusPending = false;
    resultViewport.reset(); saveResultPreferences({viewport:Nisar.captureViewport(resultViewport)});
    resultsEl("investigation").scrollIntoView({behavior:"smooth",block:"start"});
  }
});

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
      Nisar.restoreViewport(resultViewport,Nisar.view.read(resultRevision()).viewport);
    } else resultViewport.invalidate();
    if (investigationFocusPending) focusInvestigationStop();
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
  resultViewport.zoomBy(factor);
  saveResultPreferences({viewport:Nisar.captureViewport(resultViewport)});
}

function renderResultsV2(data) {
  const newResult = resultData !== data;
  resultData = data;
  if (newResult) {
    investigationFocus = null; investigationFocusPending = false;
    resultLayers = new Map();
    resultImage = resultOverlay = null;
    resultViewport.setRaster(0,0);
    const saved = Nisar.view.read(resultRevision());
    resultMode = saved.date || "t0";
    resultShowChange = !!data.images?.["overlay.png"] && (saved.showChange ?? true);
    resultGrayscale = !!data.images?.["t0_hh.png"] && !!data.images?.["t1_hh.png"] && (saved.grayscale ?? false);
    investigation.render(data);
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
  setResultText("resultEpsg", roi.epsg ? `Map reference · EPSG:${roi.epsg} ⓘ` : "Projection unavailable");
  resultsEl("resultEpsg").disabled = !roi.epsg;
  setResultText("resultMapScale", `${roi.model_spacing_m ?? "—"} m grid · EPSG:${roi.epsg ?? "—"}`);
  setResultText("resultFlaggedArea", `${resultArea(data.changed_pixels * (roi.model_spacing_m || 0) ** 2 / 1e6)} km²`);
  const pct = data.valid_pixels ? 100 * data.changed_pixels / data.valid_pixels : 0;
  setResultText("resultFlaggedPct", `${pct.toFixed(1)}% of valid model pixels · threshold ${data.threshold ?? "—"}`);
  setResultText("resultFlaggedCount", resultNumber(data.changed_pixels));
  setResultText("resultModelPct", `${pct.toFixed(1)}% flagged`);
  setResultText("resultUnflaggedPct", `${(100-pct).toFixed(1)}% not flagged`);
  resultsEl("resultSplitFill").style.width = `${Math.max(0,Math.min(100,pct))}%`;
  setResultText("resultCoverage", `${resultArea(roi.projected_rectangle_km2)} km² · EPSG:${roi.epsg ?? "—"}`);
  const sectors = [...(data.spatial_context?.zones || [])].filter(z => z.flagged).sort((a,b) => b.flagged-a.flagged).slice(0,3);
  setResultText("resultSectors", sectors.length ? sectors.map(z => z.zone.replace(/\b\w/g, c => c.toUpperCase())).join(" · ") : "No flagged sectors");
  const geo = acq.t0 || {};
  setResultText("resultGeometry", [geo.orbitPassDirection, geo.lookDirection, geo.trackNumber == null ? null : `Track ${geo.trackNumber}`, geo.frameNumber == null ? null : `Frame ${geo.frameNumber}`].filter(Boolean).join(" · ") || "Acquisition geometry unavailable");
  NisarEvidence.render(data);
  NisarExternalContext.render(resultsEl("resultExternalContext"), data);
  resultsEl("resultViewerLink").disabled = resultRevision() === null;
  showResultMode(resultMode);
}

// Result revision: the ?v= of the result image URLs (same source as /t0.png?v=… above).
function resultRevision() {
  const url = resultData?.images?.["t0.png"];
  if (!url) return null;
  const v = new URL(url, location.origin).searchParams.get("v");
  return Nisar.isResultRevision(v) ? v : null;
}
resultsEl("resultViewerLink").addEventListener("click", () => {
  const rev = resultRevision();
  if (rev === null) return;
  saveResultPreferences({viewport:Nisar.captureViewport(resultViewport)});
  location.href = `/change?v=${rev}`;
});
window.addEventListener("pagehide",() => {
  const viewport = Nisar.captureViewport(resultViewport);
  if (viewport) saveResultPreferences({viewport});
});

function saveResultPreferences(extra={}) {
  Nisar.view.save(resultRevision(), {date:resultMode,grayscale:resultGrayscale,showChange:resultShowChange,...extra});
}
document.querySelectorAll("[data-result-mode]").forEach(button => button.addEventListener("click", () => {
  resultMode = button.dataset.resultMode;
  saveResultPreferences({viewerMode:resultMode});
  showResultMode(resultMode);
}));
resultsEl("resultEpsg").addEventListener("click",() => Nisar.openProjection(resultData?.roi?.epsg));
resultsEl("resultShowChange").addEventListener("click", () => {
  resultShowChange = !resultShowChange;
  saveResultPreferences();
  showResultMode(resultMode);
});
resultsEl("resultGrayscale").addEventListener("click", () => {
  resultGrayscale = !resultGrayscale;
  saveResultPreferences();
  showResultMode(resultMode);
});
resultsEl("resultZoomIn").addEventListener("click", () => resultZoom(1.5));
resultsEl("resultZoomOut").addEventListener("click", () => resultZoom(1/1.5));
resultsEl("resultOpenReport").addEventListener("click", () => resultsEl("resultReportDialog").showModal());
resultsEl("resultCloseReport").addEventListener("click", () => resultsEl("resultReportDialog").close());
resultsEl("resultReportDialog").addEventListener("click", e => { if (e.target === e.currentTarget) e.currentTarget.close(); });

function inspectReportSector(zone) {
  const zones=["upper left","upper center","upper right","middle left","center","middle right","lower left","lower center","lower right"];
  const index=zones.indexOf(zone);
  if(index<0 || !resultViewport.width) return;
  resultViewport.scale=Math.min(resultViewport.maxScale(),resultViewport.fitScale*2.7);
  resultViewport.x=resultViewport.w/2-((index%3+.5)/3)*resultViewport.width*resultViewport.scale;
  resultViewport.y=resultViewport.h/2-((Math.floor(index/3)+.5)/3)*resultViewport.height*resultViewport.scale;
  resultViewport.clamp(); resultViewport.invalidate();
  resultsEl("resultReportDialog").close();
  resultCanvas.scrollIntoView({behavior:"smooth",block:"center"});
  saveResultPreferences({viewport:Nisar.captureViewport(resultViewport)});
}

let evidenceChatRevision = null;
let evidenceClient = null;
let evidenceRequest = null;
let evidenceHasReply = false;
const evidenceMarkdown = window.markdownit({html: false, breaks: true});
const newEvidenceClient = () => Array.from(crypto.getRandomValues(new Uint8Array(16)),
  byte => byte.toString(16).padStart(2, "0")).join("");

function evidenceMessage(label, message) {
  const log = resultsEl("evidenceChatLog");
  const line = document.createElement("div");
  line.className = "evidence-message";
  const speaker = document.createElement("strong");
  speaker.className = "evidence-speaker";
  speaker.textContent = label;
  const body = document.createElement("div");
  body.className = "evidence-message-body";
  if (label === "Qwen") body.innerHTML = evidenceMarkdown.render(message);
  else {
    body.classList.add("evidence-question");
    body.textContent = message;
  }
  line.append(speaker, body);
  log.append(line);
  while (log.childElementCount > 24) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
  return body;
}

function renderEvidenceReply(reply, text) {
  const log = resultsEl("evidenceChatLog");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
  reply.innerHTML = evidenceMarkdown.render(text);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function evidenceBusy(busy) {
  resultsEl("evidenceSend").disabled = busy;
  resultsEl("evidenceStop").hidden = !busy;
}

resultsEl("evidenceStop").addEventListener("click", () => evidenceRequest?.abort());
window.addEventListener("pagehide", () => evidenceRequest?.abort());
resultsEl("evidenceQuestion").addEventListener("keydown", event => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (!event.repeat && !evidenceRequest) {
    resultsEl("evidenceChat").requestSubmit(resultsEl("evidenceSend"));
  }
});
resultsEl("evidenceChat").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = resultsEl("evidenceQuestion");
  const question = input.value.trim();
  const data = resultData;
  const revision = resultRevision();
  if (!data || !revision || !question || evidenceRequest) return;
  evidenceMessage("You", question);
  input.value = "";
  const request = new AbortController();
  evidenceRequest = request;
  evidenceBusy(true);
  const reply = evidenceMessage("Qwen", "Preparing reply…");
  let answer = "";
  try {
    const response = await fetch("/chat", {
      method: "POST", signal: request.signal,
      headers: {"Content-Type": "application/json", "X-Nisar-Demo": "1"},
      body: JSON.stringify({revision, client: evidenceClient ??= newEvidenceClient(), question}),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      if (response.status === 404) throw new Error("This server is missing the chat endpoint. Apply the chat update and restart it.");
      throw new Error(problem.error || `Chat request failed (HTTP ${response.status}).`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "", complete = false;
    try {
      while (true) {
        const part = await reader.read();
        if (part.done) break;
        if (request.signal.aborted || evidenceRequest !== request) return;
        pending += decoder.decode(part.value, {stream: true});
        const lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) {
          if (!line) continue;
          const item = JSON.parse(line);
          if (item.error) throw new Error(item.error);
          if (item.status === "prefill" && evidenceHasReply) {
            reply.parentElement.before(Object.assign(document.createElement("p"), {
              textContent: "Qwen started a fresh context for this result.",
            }));
          }
          if (item.delta) answer += item.delta;
          if (item.done) complete = true;
        }
        if (answer) renderEvidenceReply(reply, answer);
      }
    } finally {
      if (!complete) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
    if (!complete) throw new Error("The reply was interrupted.");
    if (!answer.trim()) renderEvidenceReply(reply, "No text was generated. Try rephrasing your question.");
    evidenceHasReply = true;
  } catch (error) {
    if (evidenceRequest !== request) return;
    if (request.signal.aborted) {
      renderEvidenceReply(reply, answer || "Reply stopped.");
    } else {
      renderEvidenceReply(reply, answer || error.message);
    }
    if (answer) {
      const status = document.createElement("p");
      status.className = "evidence-reply-status";
      status.textContent = request.signal.aborted ? "Reply stopped." : error.message;
      reply.after(status);
    }
  } finally {
    if (evidenceRequest === request) {
      evidenceRequest = null;
      evidenceBusy(false);
    }
  }
});

function showResultsV2(data) {
  resultsEl("analysisPage").hidden = true;
  resultsEl("pickTab").removeAttribute("aria-current");
  resultsEl("resultsTab").setAttribute("aria-current", "page");
  resultsEl("results").hidden = false;
  renderResultsV2(data);
  const revision = resultRevision();
  if (!revision || evidenceChatRevision !== revision) {
    evidenceRequest?.abort();
    evidenceRequest = null;
    evidenceClient = null;
    evidenceHasReply = false;
    evidenceBusy(false);
    resultsEl("evidenceChatLog").replaceChildren();
    resultsEl("evidenceQuestion").value = "";
    evidenceChatRevision = revision;
  }
  resultsEl("evidenceChatHint").textContent = data.sample
    ? "Qwen uses the saved images, visible overlay footprint, and measured evidence."
    : "Qwen uses the selected before and after images, detector mask, and measured evidence.";
  NisarReport.render(data,{onInspectSector:inspectReportSector});
  resultsEl("details").textContent = JSON.stringify(data,null,2);
  resultsEl("resultSampleNotice").hidden = !data.sample;
  if (data.sample) resultsEl("resultSampleNotice").textContent = "Saved analysis sample · fixed ROI and threshold 0.10 · generated " + new Date(data.sample.generated_at).toLocaleDateString() + ". The detector was not rerun. Source filenames and checksums are included in the JSON and report exports.";
  const resultsUrl = data.sample ? `/?v=${encodeURIComponent(resultRevision())}#results` : "/#results";
  resultsEl("resultsTab").href = resultsUrl;
  document.querySelector('.app-tabs a[href^="/change"]').href = `/change?v=${encodeURIComponent(resultRevision())}`;
  resultsEl("pickPage").hidden = true;
  // "/#results" reopens this tab (used by the Results link on the Change page).
  history.replaceState(null, "", resultsUrl);
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
  const requested = new URLSearchParams(location.search).get("v");
  if (savedRun() && !requested?.startsWith("sample-")) return;
  try {
    const r = await fetch("/result" + (requested ? "?v=" + encodeURIComponent(requested) : ""), {cache: "no-store"});
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
