"use strict";
(() => {
  const el = id => document.getElementById(id);
  let sample = null, modelEnabled = true;
  function configure(capabilities) {
    modelEnabled = capabilities.model_enabled;
    sample = capabilities.sample;
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
    ["sampleLeft", "sampleTop", "sampleRight", "sampleBottom"].forEach((id, i) => { el(id).textContent = String(sample.roi[i]); });
    el("sampleSize").textContent = sample.native_size_px.map(n => n.toLocaleString("en-US")).join(" × ") + " px";
    el("sampleArea").textContent = sample.projected_area_km2.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}) + " km²";
    el("sampleThreshold").textContent = sample.threshold.toFixed(2);
    el("sampleCrs").textContent = "EPSG:" + sample.epsg;
    el("samplePreview").src = sample.preview_url;
    el("samplePreview").hidden = false;
    el("sampleStatus").textContent = "Saved " + new Date(sample.generated_at).toLocaleString(undefined, {timeZoneName:"short"}) + ". Opening results does not run any models.";
    el("sampleModeNote").textContent = modelEnabled
      ? "Optional shortcut. Use steps 1 and 2 to analyze your own files and selection."
      : "Saved sample mode is active. Live file uploads and area selection are unavailable. To run your own analysis, restart with python launch.py --mode model on a capable machine.";
  }
  function show() {
    el("samplePanel").hidden = false;
    el("pickLayout").hidden = true;
    el("newPair").hidden = el("crsTag").hidden = true;
    el("pickTitle").textContent = "Try a saved analysis";
    el("chips").replaceChildren();
    el("status").textContent = "Step 0 is optional. View the fixed sample inputs, then explore the saved results.";
    setStep(0);
  }
  function hide() {
    el("samplePanel").hidden = true;
    el("pickLayout").hidden = false;
  }
  el("sampleStep").addEventListener("click", show);
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
