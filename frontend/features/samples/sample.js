"use strict";
(() => {
  const el = id => document.getElementById(id);
  let sample = null, modelEnabled = true;
  const stamp = key => sample?.sources?.[key]?.acquisition?.zeroDopplerStartTime;
  function dateLabel(value) {
    const date = new Date(value);
    return value && Number.isFinite(date.getTime())
      ? new Intl.DateTimeFormat("en-GB", {day:"numeric", month:"long", year:"numeric", timeZone:"UTC"}).format(date)
      : "Date not recorded";
  }
  function configure(capabilities) {
    modelEnabled = capabilities.model_enabled;
    sample = capabilities.sample;
    el("appMode").hidden = false;
    el("appMode").dataset.mode = modelEnabled ? "model" : "sample";
    el("appMode").textContent = modelEnabled ? "Live model mode · GCOV uploads enabled" : "Saved sample mode · uploads unavailable";
    el("sampleAnalyzeOwn").hidden = !modelEnabled;
    el("sampleStep").disabled = false;
    const available = !!sample?.available;
    el("openSample").disabled = !available;
    if (!available) {
      el("sampleT0").textContent = "No saved analysis has been installed on this server.";
      el("sampleStatus").textContent = "The project maintainer can generate it once with python cache_sample.py.";
      return;
    }
    el("sampleT0").textContent = sample.sources.t0.name;
    el("sampleT1").textContent = sample.sources.t1.name;
    el("sampleBefore").textContent = dateLabel(stamp("t0"));
    el("sampleAfter").textContent = dateLabel(stamp("t1"));
    const days = Math.round((Date.parse(stamp("t1")) - Date.parse(stamp("t0"))) / 86400000);
    el("sampleInterval").textContent = Number.isFinite(days) ? `${days} days apart · the same geographic area at both dates` : "The same geographic area at both dates";
    el("samplePreviewCaption").firstChild.textContent = `Before · ${dateLabel(stamp("t0"))} `;
    ["sampleLeft", "sampleTop", "sampleRight", "sampleBottom"].forEach((id, i) => { el(id).textContent = String(sample.roi[i]); });
    el("sampleSize").textContent = sample.native_size_px.map(n => n.toLocaleString("en-US")).join(" × ") + " px";
    el("sampleArea").textContent = sample.projected_area_km2.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}) + " km²";
    el("sampleThreshold").textContent = sample.threshold.toFixed(2);
    el("sampleCrs").textContent = "EPSG:" + sample.epsg;
    el("samplePreview").src = sample.preview_url;
    el("samplePreview").hidden = false;
    el("sampleSavedAt").textContent = "Analysis saved " + new Date(sample.generated_at).toLocaleString(undefined, {timeZoneName:"short"}) + ". Source checksums and run provenance are included in the report exports.";
    el("sampleStatus").textContent = "Saved results · fixed dates, area, and threshold. Opening them does not run any models.";
    el("sampleModeNote").textContent = modelEnabled
      ? "Have your own NISAR files? You can also load a pair and choose an area to analyze."
      : "This server is in saved sample mode. To upload your own GCOV files, stop this server and restart with python launch.py --mode model on a capable machine.";
  }
  function show() {
    el("samplePanel").hidden = false;
    el("pickLayout").hidden = true;
    el("newPair").hidden = el("crsTag").hidden = true;
    el("pickTitle").textContent = "Explore a case study";
    el("chips").replaceChildren();
    el("status").textContent = "Start with a saved radar comparison from Karakalpakstan, Uzbekistan.";
    setStep(0);
  }
  function hide() {
    el("samplePanel").hidden = true;
    el("pickLayout").hidden = false;
  }
  el("sampleStep").addEventListener("click", show);
  el("sampleAnalyzeOwn").addEventListener("click", () => { if (modelEnabled) showUpload(true); });
  el("openSample").addEventListener("click", async () => {
    if (!sample?.available) return;
    el("openSample").disabled = true;
    try {
      const response = await fetch("/result?v=" + encodeURIComponent(sample.revision), {cache:"no-store"});
      const data = await response.json();
      if (!response.ok || !data.result) throw new Error(data.error || "Saved results are unavailable.");
      showResultsV2(data.result);
    } catch (error) {
      el("sampleStatus").textContent = error.message;
    } finally { el("openSample").disabled = false; }
  });
  window.NisarSample = {configure, show, hide};
})();
