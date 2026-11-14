"use strict";
const $ = id => document.getElementById(id);
const preview = $("preview");
const fields = ["left","top","right","bottom"].map($);
let state, photo, start, pointer, busy = false, mode = "rg", previewDate = "t0";
let requestedMode = "rg", requestedDate = "t0", previewRequest = 0;
const previews = new Map();
const tiles = new Map();
let inFlight = 0;
const view = new RasterViewport(preview, paint);
view.enableDrag(1);

function status(message) { $("status").textContent = message; }
function setStep(n) {
  for (const li of $("steps").children) {
    const s = Number(li.dataset.step);
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n);
    if (s === n) li.querySelector("button").setAttribute("aria-current", "step");
    else li.querySelector("button").removeAttribute("aria-current");
  }
}
function setBusy(value) {
  busy = value;
  for (const el of document.querySelectorAll("form input,form button,#newPair,#loadStep,#areaStep")) el.disabled = value;
  $("areaStep").disabled = value || !state;
  if (!value) updateSelection();
}
async function post(path,body=null,extra={}) {
  const headers = {"X-Nisar-Demo":"1",...extra};
  let r, raw;
  try {
    r = await fetch(path,{method:"POST",headers,body});
    raw = await r.text();
  } catch (err) {
    throw new Error(`${path}: connection interrupted (${err.message}). Check the demo terminal.`);
  }
  let data;
  try { data = JSON.parse(raw); }
  catch {
    throw new Error(`${path}: HTTP ${r.status}; ${raw.trim() ? "server returned non-JSON" : "empty response"}. Check the demo terminal.`);
  }
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} — ${data.error || "request failed"}`);
  return data;
}
function showProgress(received,total) {
  $("transferProgress").max = total;
  $("transferProgress").value = received;
  $("transferPercent").textContent = `${Math.floor(100*received/total)}%`;
}
async function uploadFile(name,file,onProgress) {
  const chunk = 8 * 1024 * 1024;
  if (!file.size) throw new Error(`${name.toUpperCase()} is empty.`);
  for (let offset=0; offset<file.size;) {
    const end = Math.min(offset+chunk,file.size);
    status(`Uploading ${name.toUpperCase()}: ${(100*offset/file.size).toFixed(1)}% (${file.name})…`);
    const ack = await post("/upload/"+name,file.slice(offset,end),{
      "Content-Type":"application/octet-stream",
      "X-Upload-Offset":String(offset),
      "X-Upload-Size":String(file.size)
    });
    if (ack.received !== end || ack.complete !== (end === file.size)) {
      throw new Error(`${name.toUpperCase()}: invalid upload acknowledgement. Restart the demo and refresh the page.`);
    }
    offset = end;
    onProgress(offset);
  }
}
function setBox(box) {
  fields.forEach((f,i) => { f.value = box[i]; });
  draw();
}
function draw() { view.invalidate(); updateSelection(); }

const fmt = n => n.toLocaleString("en-US");
function currentBox() {
  const box = fields.map(f => f.valueAsNumber);
  return box.every(Number.isFinite) && box[2] > box[0] && box[3] > box[1] ? box : null;
}
function updateSelection() {
  const box = state && currentBox();
  const ids = ["selPx","selCells","selArea","selShare","selBox"];
  if (!box) {
    ids.forEach(id => { $(id).textContent = "—"; });
    $("runBtn").disabled = true;
    $("runNote").textContent = state ? "Draw a selection to enable the model run." : "Load a pair first.";
    return;
  }
  const [x0,y0,x1,y1] = box, w = x1-x0, h = y1-y0;
  const [dx,dy] = state.native_spacing_m.map(Math.abs);
  const [ox,oy,OX,OY] = state.overlap;
  const inside = ox<=x0 && oy<=y0 && x1<=OX && y1<=OY;
  $("selPx").textContent = `${fmt(w)} × ${fmt(h)} px`;
  $("selCells").textContent = `≈ ${fmt(Math.floor(w*dx/state.model_spacing_m))} × ${fmt(Math.floor(h*dy/state.model_spacing_m))} cells`;
  $("selArea").textContent = `${(w*h*dx*dy/1e6).toFixed(2)} km²`;
  $("selShare").textContent = `${(100*w*h/((OX-ox)*(OY-oy))).toFixed(1)} %`;
  $("selBox").textContent = `${x0}, ${y0} → ${x1}, ${y1}`;
  const validThreshold = $("thresholdValue").validity.valid;
  $("runBtn").disabled = busy || !inside || !validThreshold;
  $("runNote").textContent = !validThreshold ? "Enter a threshold from 0.01 to 0.99."
    : inside ? "Ready to compare this area at both dates."
                                    : "Selection must stay inside the dashed common footprint.";
}

function paint(ctx) {
  if (!photo || !state) return;
  const s = state.stride;
  ctx.save();
  ctx.beginPath(); ctx.rect(0,0,state.width,state.height); ctx.clip();
  ctx.drawImage(photo,0,0,photo.naturalWidth*s,photo.naturalHeight*s);
  paintTiles(ctx);
  ctx.restore();
  const [x,y,X,Y] = state.overlap;
  ctx.save();
  ctx.strokeStyle = "white"; ctx.lineWidth = 1.5/view.scale;
  ctx.setLineDash([6/view.scale,5/view.scale]);
  ctx.strokeRect(x,y,X-x,Y-y);
  ctx.restore();
  const box = currentBox();
  if (!box) return;
  const [a,b,c,d] = box;
  ctx.fillStyle = "rgba(255,200,30,0.2)"; ctx.strokeStyle = "#ffc81e";
  ctx.lineWidth = 2/view.scale;
  ctx.fillRect(a,b,c-a,d-b);
  ctx.strokeRect(a,b,c-a,d-b);
  // Corner handles and a size label, drawn at screen scale.
  const k = 1/view.scale, hs = 6*k;
  ctx.fillStyle = "#fff";
  for (const [px,py] of [[a,b],[c,b],[a,d],[c,d]]) ctx.fillRect(px-hs/2,py-hs/2,hs,hs);
  const text = `Selection · ${fmt(c-a)} × ${fmt(d-b)} px`;
  ctx.font = `600 ${12*k}px system-ui, sans-serif`;
  const tw = ctx.measureText(text).width + 12*k;
  ctx.fillStyle = "#ffc81e";
  ctx.fillRect(a,b-20*k,tw,18*k);
  ctx.fillStyle = "#14212b";
  ctx.fillText(text,a+6*k,b-7*k);
}
function paintTiles(ctx) {
  const step = 2**Math.max(0,Math.floor(Math.log2(1/view.scale)));
  if (step >= state.stride) return;
  const span = step*state.preview_tile_size;
  const [x,y,X,Y] = view.visible();
  const needed = [];
  for (let row=Math.floor(y/span); row<Math.ceil(Y/span); row++) {
    for (let col=Math.floor(x/span); col<Math.ceil(X/span); col++) {
      const url = `/preview-tile.png?v=${state.revision}&date=${previewDate}&mode=${mode}&step=${step}&col=${col}&row=${row}`;
      const tile = tiles.get(url);
      if (tile) {
        tiles.delete(url); tiles.set(url,tile);
        if (tile.ready) {
          const w = tile.image.naturalWidth*step, h = tile.image.naturalHeight*step;
          ctx.clearRect(col*span,row*span,w,h);
          ctx.drawImage(tile.image,col*span,row*span,w,h);
        }
      } else {
        needed.push({url, distance:Math.hypot((col+.5)*span-(x+X)/2,(row+.5)*span-(y+Y)/2)});
      }
    }
  }
  needed.sort((a,b) => a.distance-b.distance);
  for (const {url} of needed.slice(0,Math.max(0,4-inFlight))) {
    const image = new Image(), tile = {image,ready:false};
    tiles.set(url,tile);
    inFlight++;
    image.src = url;
    image.decode().then(() => { tile.ready = true; }, () => {
      // Keep the overview visible if a tile is unavailable.
    }).finally(() => { inFlight--; view.invalidate(); });
  }
  while (tiles.size > 64) tiles.delete(tiles.keys().next().value);
}
function point(e) {
  const [x,y,X,Y] = state.overlap;
  const [nx,ny] = view.pixel(...view.local(e));
  return [Math.max(x,Math.min(X,nx)),Math.max(y,Math.min(Y,ny))];
}
preview.addEventListener("pointerdown",e => {
  if (busy || !state || !e.isPrimary || e.button !== 0) return;
  start = point(e); pointer = e.pointerId;
  preview.setPointerCapture(pointer);
  setBox([Math.floor(start[0]),Math.floor(start[1]),Math.ceil(start[0]),Math.ceil(start[1])]);
});
function drag(e) {
  if (!start || e.pointerId !== pointer) return;
  const [x,y] = point(e);
  setBox([Math.floor(Math.min(x,start[0])),Math.floor(Math.min(y,start[1])),
          Math.ceil(Math.max(x,start[0])),Math.ceil(Math.max(y,start[1]))]);
}
preview.addEventListener("pointermove",e => {
  drag(e);
  if (!state) return;
  const [x,y] = view.pixel(...view.local(e));
  const ok = x>=0 && y>=0 && x<state.width && y<state.height;
  $("readout").innerHTML = ok ? `x <b>${fmt(Math.floor(x))}</b> · y <b>${fmt(Math.floor(y))}</b> px · zoom <b>${(view.scale/view.fitScale).toFixed(1)}×</b>`
                              : "Cursor <b>—</b>";
});
preview.addEventListener("pointerleave",() => { $("readout").innerHTML = "Cursor <b>—</b>"; });
preview.addEventListener("pointerup",e => {
  if (e.pointerId === pointer) { drag(e); start = null; pointer = null; }
});
preview.addEventListener("lostpointercapture",e => {
  if (e.pointerId === pointer) { start = null; pointer = null; }
});
fields.forEach(f => f.addEventListener("input",draw));

$("zoomIn").addEventListener("click",() => view.zoomBy(1.5));
$("zoomOut").addEventListener("click",() => view.zoomBy(1/1.5));
$("zoomFit").addEventListener("click",() => view.fit());
$("clearBox").addEventListener("click",() => { fields.forEach(f => { f.value = ""; }); draw(); });

for (const b of document.querySelectorAll("[data-method]")) {
  b.addEventListener("click",() => {
    const coords = b.dataset.method === "coords";
    for (const o of document.querySelectorAll("[data-method]")) o.setAttribute("aria-pressed",String(o === b));
    $("coords").hidden = !coords;
    $("drawHint").hidden = coords;
  });
}

async function loadPreview(data,mode,date="t0") {
  const url = date === "t1" ? (mode === "hh" ? data.preview_t1_hh_url : data.preview_t1_url)
    : (mode === "hh" ? data.preview_hh_url : data.preview_url);
  if (!url) throw new Error("Reload the pair to prepare previews for both dates.");
  if (!previews.has(url)) {
    const img = new Image();
    img.src = url;
    previews.set(url,img.decode().then(() => img).catch(err => { previews.delete(url); throw err; }));
  }
  return previews.get(url);
}
function showMode(m) {
  for (const b of document.querySelectorAll("[data-mode]")) b.setAttribute("aria-pressed",String(b.dataset.mode === m));
  $("legendRamp").innerHTML = m === "hh" ? '<span class="sw sw-hh"></span>HH backscatter (dB)'
                                         : '<span class="sw sw-rg"></span>HH red · HV green';
}
function showPreviewDate() {
  for (const b of document.querySelectorAll("[data-date]")) b.setAttribute("aria-pressed",String(b.dataset.date === previewDate));
  const label = previewDate === "t0" ? "Before · T0" : "After · T1";
  const stamp = state?.acquisitions?.[previewDate]?.zeroDopplerStartTime;
  $("previewDate").textContent = label + (stamp ? ` · ${when(stamp)}` : "");
  preview.setAttribute("aria-label",`${label} radar preview · ${mode === "hh" ? "HH grayscale" : "HH/HV colour"} · selection on the shared T0 grid`);
}
async function switchPreview(date,m) {
  if (!state || (date === previewDate && m === mode && requestedDate === date && requestedMode === m)) return;
  const pair = state, request = ++previewRequest;
  requestedDate = date; requestedMode = m;
  $("previewStatus").hidden = false;
  $("previewStatus").textContent = `Loading ${date.toUpperCase()} preview…`;
  try {
    const img = await loadPreview(pair,m,date);
    if (state !== pair || request !== previewRequest) return;
    photo = img; mode = m; previewDate = date;
    showMode(mode); showPreviewDate();
    $("previewStatus").hidden = true;
    draw(); // Preserve the viewport and the selection when changing dates or tones.
  } catch (err) {
    if (state !== pair || request !== previewRequest) return;
    requestedDate = previewDate; requestedMode = mode;
    $("previewStatus").textContent = "Could not switch preview: "+err.message;
  }
}
for (const b of document.querySelectorAll("[data-mode]")) b.addEventListener("click",() => switchPreview(requestedDate,b.dataset.mode));
for (const b of document.querySelectorAll("[data-date]")) b.addEventListener("click",() => switchPreview(b.dataset.date,requestedMode));

function when(t) {
  return typeof t === "string" ? t.slice(0,16).replace("T"," ") + " UTC" : "Time not in metadata";
}
function orbitText(m) {
  const parts = [];
  if (m.orbitPassDirection) parts.push(m.orbitPassDirection[0].toUpperCase()+m.orbitPassDirection.slice(1));
  if (m.lookDirection) parts.push(m.lookDirection+"-looking");
  if (m.trackNumber !== undefined) parts.push("track "+m.trackNumber);
  if (m.frameNumber !== undefined) parts.push("frame "+m.frameNumber);
  if (m.absoluteOrbitNumber !== undefined) parts.push("orbit "+m.absoluteOrbitNumber);
  return parts.join(" · ") || "No orbit metadata";
}
function chip(label,value) {
  const span = document.createElement("span"), b = document.createElement("b");
  b.textContent = value;
  span.append(label+" ",b);
  return span;
}
function showPairDetails(p) {
  const m0 = p.acquisitions?.t0 || {}, m1 = p.acquisitions?.t1 || {};
  const [ox,oy,OX,OY] = p.overlap;
  const [dx,dy] = p.native_spacing_m.map(Math.abs);
  $("crsTag").textContent = "EPSG:"+p.epsg; $("crsTag").hidden = false;
  const frequency = document.createElement("button");
  frequency.type = "button"; frequency.className = "frequency-info-button";
  frequency.textContent = `Frequency ${p.frequency} ⓘ`;
  frequency.setAttribute("aria-label", `About Frequency ${p.frequency}`);
  frequency.setAttribute("aria-haspopup", "dialog");
  frequency.addEventListener("click", () => $("frequencyDialog").showModal());
  $("frequencyTitle").textContent = `About Frequency ${p.frequency}`;
  $("frequencyExplanation").textContent = `Frequency ${p.frequency} identifies the set of radar measurements inside your GCOV file that this model uses for change detection.`;
  $("chips").replaceChildren(
    chip("T0",when(m0.zeroDopplerStartTime)), chip("T1",when(m1.zeroDopplerStartTime)),
    frequency, chip("Grid",`${fmt(p.width)} × ${fmt(p.height)} px`));
  $("t0Date").textContent = when(m0.zeroDopplerStartTime);
  $("t0Meta").textContent = orbitText(m0);
  $("t1Date").textContent = when(m1.zeroDopplerStartTime);
  $("t1Meta").textContent = orbitText(m1);
  $("overlapSize").textContent = `${((OX-ox)*(OY-oy)*dx*dy/1e6).toFixed(1)} km²`;
  $("overlapMeta").textContent = `${fmt(OX-ox)} × ${fmt(OY-oy)} native px · x ${ox}–${OX}, y ${oy}–${OY}`;
  $("freqPill").textContent = "Frequency "+p.frequency+" ⓘ";
  $("nativeSpacing").textContent = `${dx} × ${dy} m`;
  $("modelSpacing").textContent = `${p.model_spacing_m} m`;
  $("analysisResolution").textContent = `The model compares ${p.model_spacing_m} × ${p.model_spacing_m} m cells. Use the threshold above to decide which possible changes are flagged.`;
  $("gridExplanation").textContent = dx === p.model_spacing_m && dy === p.model_spacing_m
    ? "Each analysis cell matches one image pixel. Both dates are compared on the same map grid."
    : `Image pixels are averaged into ${p.model_spacing_m} × ${p.model_spacing_m} m analysis cells so both dates match the model's training resolution.`;
  $("gridSize").textContent = `${fmt(p.width)} × ${fmt(p.height)} px`;
  $("crs").textContent = "EPSG:"+p.epsg;
  const list = $("warnings");
  list.replaceChildren();
  for (const w of p.warnings.length ? p.warnings : ["Acquisition geometry matches; the pair is comparable."]) {
    const li = document.createElement("li");
    li.textContent = w;
    if (!p.warnings.length) li.className = "ok";
    list.append(li);
  }
}

function showUpload(open) {
  $("uploadPanel").hidden = !open;
  $("cancelUpload").hidden = !state;
  $("infer").hidden = open;
  $("mapUi").hidden = $("pairCard").hidden = open || !state;
  $("newPair").hidden = open || !state;
  $("crsTag").hidden = open || !state;
  $("pickLayout").classList.toggle("loading-pair", open);
  $("pickTitle").textContent = open ? "Load GCOV images" : "Pick a change area";
  setStep(open ? 1 : 2);
  if (!busy) status(open ? "Choose two L-SAR GCOV .h5 files." : "Compare Before and After, then draw an area covered by both dates.");
}
$("newPair").addEventListener("click",() => showUpload(true));
$("cancelUpload").addEventListener("click",() => showUpload(false));
$("loadStep").addEventListener("click",() => showUpload(true));
$("areaStep").addEventListener("click",() => { if (state) showUpload(false); });
$("analysisInfoBtn").addEventListener("click",() => $("analysisInfoDialog").showModal());
$("freqPill").addEventListener("click",() => $("frequencyDialog").showModal());
for (const button of document.querySelectorAll("[data-close-dialog]")) {
  button.addEventListener("click",() => $(button.dataset.closeDialog).close());
}
for (const id of ["analysisInfoDialog","frequencyDialog"]) {
  $(id).addEventListener("click", e => { if (e.target === e.currentTarget) e.currentTarget.close(); });
}

function updateThreshold(fromNumber=false) {
  if (fromNumber) {
    const valid = $("thresholdValue").validity.valid;
    $("thresholdError").hidden = valid;
    $("thresholdValue").setAttribute("aria-invalid", String(!valid));
    if (!valid) { updateSelection(); return; }
    $("threshold").value = $("thresholdValue").value;
  } else {
    $("thresholdValue").value = $("threshold").valueAsNumber.toFixed(2);
    $("thresholdError").hidden = true;
    $("thresholdValue").setAttribute("aria-invalid", "false");
  }
  const value = $("threshold").valueAsNumber;
  const hint = value < .35 ? "More sensitive · includes lower-scoring changes."
    : value < .7 ? "Balanced · keeps moderate and higher-scoring changes."
    : "More selective · keeps only higher-scoring changes.";
  $("thresholdHint").textContent = hint;
  $("threshold").setAttribute("aria-valuetext", `${value.toFixed(2)}. ${hint}`);
  updateSelection();
}
$("threshold").addEventListener("input", () => updateThreshold());
$("thresholdValue").addEventListener("input", () => updateThreshold(true));
$("thresholdValue").addEventListener("change", () => {
  if ($("thresholdValue").validity.valid) $("thresholdValue").value = $("threshold").valueAsNumber.toFixed(2);
});
updateThreshold();

async function showPair(data) {
    ++previewRequest; previews.clear();
    requestedMode = "rg"; requestedDate = "t0"; showMode("rg");
    const img = await loadPreview(data,"rg");
    state = data; photo = img; mode = "rg"; previewDate = "t0"; tiles.clear();
    showPreviewDate(); $("previewStatus").hidden = true;
    fields.forEach(f => { f.value = ""; });
    showPairDetails(state);
    $("threshold").value = String(data.threshold ?? .1);
    updateThreshold();
    $("mapUi").hidden = $("pairCard").hidden = $("newPair").hidden = false;
    showUpload(false);
    view.setRaster(state.width,state.height);
    setStep(2);
    updateSelection();
    status("Compare Before and After, then draw an area covered by both dates.");
}

setStep(1);
updateSelection();
(async () => {
  setBusy(true);
  try {
    const runId = savedRun();
    if (runId) {
      try { await followRun(runId); }
      catch (err) { status(err.message); }
    }
    const r = await fetch("/pair");
    if (r.ok) {
      const {pair} = await r.json();
      if (pair) await showPair(pair);
      else showUpload(true);
    }
  } catch (err) { status("Could not load the current pair: "+err.message); }
  finally { setBusy(false); }
})();

$("upload").addEventListener("submit",async e => {
  e.preventDefault();
  if (busy) return;
  const files = [$("t0").files[0],$("t1").files[0]];
  if (files.some(f => !f || !f.size || !/\.(h5|hdf5)$/i.test(f.name))) {
    status("Choose a non-empty .h5 or .hdf5 file for each date.");
    return;
  }
  state = null; photo = null; $("results").hidden = $("pairCard").hidden = $("mapUi").hidden = true;
  $("cancelUpload").hidden = true;
  view.setRaster(0,0);
  setStep(1);
  const total = files[0].size + files[1].size;
  $("uploadProgress").hidden = false;
  showProgress(0,total);
  setBusy(true);
  try {
    let sent = 0;
    for (const [i,name] of ["t0","t1"].entries()) {
      await uploadFile(name,files[i],received => showProgress(sent+received,total));
      sent += files[i].size;
    }
    status("Both files uploaded. Preparing Before and After previews…");
    await showPair(await post("/prepare"));
  } catch (err) { status(err.message); }
  finally { setBusy(false); $("uploadProgress").hidden = true; }
});

function savedRun(value) {
  try {
    if (value === null) sessionStorage.removeItem("nisar-active-run");
    else if (value !== undefined) sessionStorage.setItem("nisar-active-run", value);
    return sessionStorage.getItem("nisar-active-run");
  } catch { return null; }
}

function showAnalysisPage() {
  $("pickPage").hidden = $("results").hidden = true;
  $("analysisPage").hidden = false;
  $("modelProgress").setAttribute("aria-busy", "true");
  $("resultsTab").removeAttribute("aria-current");
  $("pickTab").setAttribute("aria-current", "page");
  $("analysisTitle").textContent = "Looking for changes in your area";
  $("analysisBadge").textContent = "Analysis in progress";
  $("analysisBack").hidden = true;
  $("analysisFootnote").textContent = "Results will open automatically. You can reload this tab and reconnect to the running analysis.";
  const box = state && currentBox();
  $("analysisSelection").textContent = box
    ? `${$("selArea").textContent} selected · Before (T0) compared with After (T1) · threshold ${$("threshold").valueAsNumber.toFixed(2)}`
    : "Reconnecting to the analysis already running on the server.";
  window.scrollTo(0,0);
  $("analysisTitle").focus({preventScroll:true});
}
function showAnalysisError(err) {
  $("analysisTitle").textContent = "Your analysis could not finish";
  $("analysisBadge").textContent = "Analysis interrupted";
  $("modelProgressTitle").textContent = "Analysis interrupted";
  $("modelProgressText").textContent = err.message;
  $("modelStageProgress").hidden = true;
  $("modelProgress").setAttribute("aria-busy", "false");
  $("analysisBack").hidden = false;
  $("analysisFootnote").textContent = "Return to your area selection to review the inputs and try again.";
}
$("analysisBack").addEventListener("click",() => showPickTab());

function updateModelProgress(job) {
  const stages = ["read", "detect", "classify", "report", "render"];
  const index = job.status === "complete" ? stages.length : stages.indexOf(job.stage);
  for (const li of document.querySelectorAll("[data-stage]")) {
    const i = stages.indexOf(li.dataset.stage);
    li.classList.toggle("done", i < index);
    li.classList.toggle("active", i === index);
    if (i === index) li.setAttribute("aria-current", "step");
    else li.removeAttribute("aria-current");
  }
  const progress = $("modelStageProgress");
  let message = job.message;
  if (job.total > 0 && Number.isFinite(job.completed)) {
    progress.max = job.total;
    progress.value = job.completed;
    message += ` · ${Math.floor(100 * job.completed / job.total)}%`;
  } else {
    progress.removeAttribute("value");
  }
  if (job.status === "complete") { progress.max = progress.value = 1; }
  if (job.status === "complete") $("modelProgressTitle").textContent = "Analysis complete";
  $("modelProgressText").textContent = message;
  $("runBtn").textContent = ({read:"Reading images…", detect:"Detecting changes…", classify:"Measuring changes…", report:"Generating explanation…", render:"Preparing results…"})[job.stage] || "Analyzing…";
  status(message);
}

async function followRun(runId) {
  savedRun(runId);
  showAnalysisPage();
  $("modelProgressTitle").textContent = "Analyzing your area";
  $("modelStageProgress").hidden = false;
  $("modelProgress").setAttribute("aria-busy", "true");
  let started = Date.now();
  const tick = () => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    $("modelElapsed").textContent = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  let finished = false;
  try {
    while (true) {
      let response;
      try { response = await fetch(`/infer/${encodeURIComponent(runId)}`, {cache:"no-store"}); }
      catch {
        $("modelProgressText").textContent = "Connection interrupted. Reconnecting to the running analysis…";
        await new Promise(resolve => setTimeout(resolve, 2000));
        continue;
      }
      const job = await response.json();
      if (!response.ok || job.status === "failed") {
        finished = true;
        throw new Error(job.error || "Could not read analysis progress.");
      }
      started = Date.now() - job.elapsed_s * 1000;
      tick();
      updateModelProgress(job);
      if (job.status === "complete") {
        finished = true;
        showResultsV2(job.result);
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 750));
    }
  } catch (err) {
    showAnalysisError(err);
    throw err;
  } finally {
    clearInterval(timer);
    if (finished) savedRun(null);
    $("modelProgress").setAttribute("aria-busy", "false");
    $("runBtn").textContent = "Run model on selection →";
  }
}

$("infer").addEventListener("submit",async e => {
  e.preventDefault();
  if (busy || !state || !currentBox()) return;
  setBusy(true);
  showAnalysisPage();
  $("modelProgressTitle").textContent = "Analyzing your area";
  $("modelStageProgress").hidden = false;
  updateModelProgress({stage:"read", message:"Starting analysis…"});
  try {
    const job = await post("/infer",JSON.stringify({box:fields.map(f=>f.valueAsNumber),revision:state.revision,
      threshold:$("threshold").valueAsNumber}),{"Content-Type":"application/json"});
    await followRun(job.run_id);
  } catch (err) {
    status(err.message);
    showAnalysisError(err);
  }
  finally { $("runBtn").textContent = "Run model on selection →"; setBusy(false); }
});
