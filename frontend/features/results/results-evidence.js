"use strict";

(() => {
  const finite = n => typeof n === "number" && Number.isFinite(n);
  const number = (n,d=0) => finite(n) ? n.toLocaleString("en-US",{maximumFractionDigits:d}) : "Not recorded";
  const signed = n => finite(n) ? `${n > 0 ? "+" : ""}${n.toFixed(2)}` : "Not enough samples";
  const percent = n => finite(n) ? `${n.toFixed(1)}%` : "Not recorded";
  const title = value => String(value || "").replace(/\b\w/g,c => c.toUpperCase());
  let data, channel = 0, exportUrls = [];

  function metrics(result) {
    const roi = result.roi || {}, spacing = roi.model_spacing_m;
    const cells = finite(roi.model_grid_width_px) && finite(roi.model_grid_height_px)
      ? roi.model_grid_width_px*roi.model_grid_height_px : null;
    const areaPerCell = finite(spacing) && spacing > 0 ? spacing**2/1e6 : null;
    const valid = result.valid_pixels, flags = result.changed_pixels;
    return {
      cells,
      coveragePct: finite(valid) && cells > 0 ? 100*valid/cells : null,
      validArea: finite(valid) && areaPerCell !== null ? valid*areaPerCell : null,
      flaggedArea: finite(flags) && areaPerCell !== null ? flags*areaPerCell : null,
      unflaggedArea: finite(valid) && finite(flags) && areaPerCell !== null ? (valid-flags)*areaPerCell : null,
    };
  }

  function node(tag,text,className) {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = String(text);
    if (className) result.className = className;
    return result;
  }
  const byId = id => document.getElementById(id);
  function table(headers,rows,caption) {
    const wrap = node("div",undefined,"evidence-table-wrap");
    const t = node("table",undefined,"evidence-table");
    if (caption) t.append(node("caption",caption));
    const head = node("thead"), tr = node("tr");
    for (const h of headers) { const th = node("th",h); th.scope = "col"; tr.append(th); }
    head.append(tr); t.append(head);
    const body = node("tbody");
    for (const row of rows) {
      const line = node("tr");
      row.forEach((value,i) => {
        const td = node(i ? "td" : "th");
        if (!i) td.scope = "row";
        if (value instanceof Node) td.append(value); else td.textContent = String(value ?? "Not recorded");
        line.append(td);
      });
      body.append(line);
    }
    t.append(body); wrap.append(t); return wrap;
  }
  function definition(rows) {
    const list = node("dl",undefined,"science-definitions");
    for (const [label,value] of rows) {
      const row = node("div"); row.append(node("dt",label),node("dd",value ?? "Not recorded")); list.append(row);
    }
    return list;
  }

  function renderPatterns() {
    if (!data) return;
    document.querySelectorAll("[data-evidence-channel]").forEach(b => b.setAttribute("aria-pressed",String(Number(b.dataset.evidenceChannel) === channel)));
    const root = byId("resultPatternComparisons"); root.replaceChildren();
    const pairs = data.paired_sector_deltas || [];
    const usable = pairs.filter(row => finite(row.flagged?.median_delta_hh_hv_db?.[channel]));
    const name = ["HH","HV"][channel];
    byId("resultPattern").textContent = usable.length
      ? `Compare flagged cells with other valid cells in the ${usable.length} leading sectors. ${name} is ${channel ? "a different receive polarization from the transmitted wave" : "the same transmit and receive polarization"}.`
      : data.changed_pixels === 0 ? "No changes were flagged at this threshold."
      : "Sector comparisons were not recorded or had too few valid cells to report.";
    const values = pairs.flatMap(row => [row.flagged?.median_delta_hh_hv_db?.[channel],row.unflagged?.median_delta_hh_hv_db?.[channel]]).filter(finite);
    const scale = Math.max(1,...values.map(Math.abs));
    for (const pair of pairs) {
      const f = pair.flagged?.median_delta_hh_hv_db?.[channel];
      const b = pair.unflagged?.median_delta_hh_hv_db?.[channel];
      const card = node("section",undefined,"sector-comparison");
      const heading = node("div",undefined,"sector-comparison-head");
      heading.append(node("h4",title(pair.zone)),node("span",finite(f) ? `${f < 0 ? "↓ Weaker" : f > 0 ? "↑ Stronger" : "No median shift"} ${name} return` : "Limited samples"));
      card.append(heading);
      for (const [label,group,value] of [["Flagged",pair.flagged,f],["Other valid cells",pair.unflagged,b]]) {
        const row = node("div",undefined,"signal-row");
        const labelEl = node("div",label); labelEl.append(node("small",`${number(group?.cells)} cells`));
        const bar = node("div",undefined,"signal-track"); bar.setAttribute("aria-hidden","true");
        const fill = node("span",undefined, label === "Flagged" ? "flagged" : "background");
        if (finite(value)) {
          fill.style.left = `${50 + Math.min(0,value/scale*48)}%`;
          fill.style.width = `${Math.abs(value)/scale*48}%`;
        }
        bar.append(fill); row.append(labelEl,bar,node("strong",finite(value) ? `${signed(value)} dB` : "n/a")); card.append(row);
      }
      const contrast = finite(f) && finite(b) ? f-b : null;
      card.append(node("p",contrast !== null ? `Flagged − other cells: ${signed(contrast)} dB difference between group medians.`
        : "A group needs at least 32 valid cells to report its median.","sector-contrast"));
      root.append(card);
    }
    byId("resultPatternNote").textContent = "Medians of cell-by-cell T1 − T0 changes on the model grid. Sectors refer to the image frame. Stronger or weaker returns alone do not establish an environmental cause.";
  }

  function renderScience() {
    const roi = data.roi || {}, settings = data.analysis_settings || {}, m = metrics(data);
    const root = byId("scienceMetrics"); root.replaceChildren();
    const cards = [
      ["Usable model coverage",percent(m.coveragePct),`${number(data.valid_pixels)} / ${number(m.cells)} grid cells valid at both dates`],
      ["Area with valid model data",finite(m.validArea) ? `${number(m.validArea,2)} km²` : "Not recorded","The denominator used for the flagged percentage"],
      ["Valid area without flags",finite(m.unflaggedArea) ? `${number(m.unflaggedArea,2)} km²` : "Not recorded","Not flagged does not mean confirmed stable"],
      ["Analysis cell size",finite(roi.model_spacing_m) ? `${number(roi.model_spacing_m)} × ${number(roi.model_spacing_m)} m` : "Not recorded","Grid spacing, not a minimum detectable change size"],
    ];
    for (const [label,value,help] of cards) {
      const card = node("div",undefined,"science-metric"); card.append(node("span",label),node("strong",value),node("small",help)); root.append(card);
    }
    const warnings = byId("scienceWarnings"); warnings.replaceChildren();
    if (data.warnings?.length) {
      for (const warning of data.warnings) warnings.append(node("p",warning,"science-warning"));
    } else warnings.append(node("p","No comparability mismatch was reported in the available metadata. Review missing fields and local conditions before attributing a cause.","science-check"));

    const medians = data.roi_measurements?.median_hh_hv_db_t0_t1 || [];
    byId("scienceBackscatter").replaceChildren(table(["Channel","Before (dB)","After (dB)","Δ medians (dB)"], [0,1].map(i => {
      const before = medians[0]?.[i], after = medians[1]?.[i];
      return [["HH","HV"][i],finite(before) ? signed(before) : "Not recorded",finite(after) ? signed(after) : "Not recorded",finite(before) && finite(after) ? signed(after-before) : "Not recorded"];
    }),"Whole-selection native-pixel medians"),node("p","These are differences between date medians. The sector comparisons above use medians of paired pixel changes, a different statistic.","science-note"));

    const zones = data.spatial_context?.zones || [];
    const sectorRoot = byId("scienceSectors"); sectorRoot.replaceChildren();
    const grid = node("div",undefined,"science-sector-grid");
    for (const z of zones) {
      const pct = z.valid > 0 ? 100*z.flagged/z.valid : null;
      const cell = node("div",undefined,"science-sector");
      if (pct !== null) cell.style.background = `rgba(13,92,99,${.04+.25*pct/100})`;
      cell.append(node("span",title(z.zone)),node("strong",pct !== null ? percent(pct) : "No valid data"));
      cell.title = `${number(z.flagged)} flagged / ${number(z.valid)} valid model cells`;
      grid.append(cell);
    }
    sectorRoot.append(grid,node("p","Flagged share of valid cells in each image sector. Locations are relative to the image, not compass directions.","science-note"));
    if (!zones.length) sectorRoot.prepend(node("p","Sector measurements were not recorded."));

    const a0 = data.acquisitions?.t0 || {}, a1 = data.acquisitions?.t1 || {};
    const fields = [["Acquisition UTC","zeroDopplerStartTime"],["Orbit direction","orbitPassDirection"],["Look direction","lookDirection"],["Track","trackNumber"],["Frame","frameNumber"],["Absolute orbit","absoluteOrbitNumber"],["Relative orbit","relativeOrbitNumber"]];
    const acq = byId("scienceAcquisitions");
    acq.replaceChildren(table(["Metadata","Before (T0)","After (T1)"],fields.map(([label,key]) => [label,a0[key] ?? "Not recorded",a1[key] ?? "Not recorded"])));
    const crs = node("button",`EPSG:${roi.epsg ?? "—"} · ${Nisar.crsName(roi.epsg)} ⓘ`,"crs-help");
    crs.type = "button"; crs.disabled = !roi.epsg; crs.addEventListener("click",() => Nisar.openProjection(roi.epsg)); acq.append(crs);
    acq.append(definition([
      ["Projected rectangle area",finite(roi.projected_rectangle_km2) ? `${number(roi.projected_rectangle_km2,2)} km²` : "Not recorded"],
      ["Native jointly valid pixels",number(roi.native_joint_valid_pixels)],
      ["Native pixel spacing",roi.native_spacing_m?.map(n => `${number(Math.abs(n),2)} m`).join(" × ")],
      ["T0 / T1 pixel boxes",[roi.t0_pixel_box,roi.t1_pixel_box].map(b => b?.join(", ") || "Not recorded").join(" / ")],
      ["Bounds (west, south, east, north), metres",roi.projected_bounds_wsen?.map(n => number(n,2)).join(" · ")],
      ["Median RTC gamma-to-sigma factors, T0 / T1",data.roi_measurements?.median_rtc_gamma_to_sigma_factor_t0_t1?.map(n => number(n,6)).join(" / ")],
    ]));
    acq.append(node("p","RTC factors are optional metadata summaries. This analysis does not apply a gamma-to-sigma conversion.","science-note"));

    const methods = byId("scienceMethods");
    methods.replaceChildren(definition([
      ["Detection threshold",finite(data.threshold) ? String(data.threshold) : "Not recorded"],
      ["Frequency group / channels",`${settings.frequency ?? "Not recorded"} / ${settings.channels?.join(", ") || "Not recorded"}`],
      ["Model grid",`${number(roi.model_grid_width_px)} × ${number(roi.model_grid_height_px)} cells`],
      ["Detector tile / overlap",`${number(settings.detector_tile_size_px)} / ${number(settings.detector_overlap_px)} model pixels`],
      ["Batch size",number(settings.batch_size)],
      ["Detector checkpoint SHA-256",settings.detector_sha256],
      ["Input dB clipping",settings.db_clip?.join(" to ")],
      ["Input preparation",settings.preprocessing],
      ["Result time",finite(data.timing_s?.total) ? `${number(data.timing_s.total,1)} s` : "Not recorded"],
    ]));
    methods.append(node("p","Only jointly valid HH/HV cells at both dates are counted. Flagged area = flagged model cells × cell area; all displayed areas are projected grid areas. Threshold scores are experimental and uncalibrated.","science-note"));
    const full = data.full_scene_context;
    if (full) methods.append(node("p",`Background context: ${number(full.sample_pairs)} matched samples at stride ${number(full.sample_stride)} native pixels. This sample is not full-footprint change detection.`,"science-note"));
    methods.append(table(["Image sector","Valid cells","Flagged cells","Flagged share"],zones.map(z => [title(z.zone),number(z.valid),number(z.flagged),z.valid ? percent(100*z.flagged/z.valid) : "n/a"]),"Detector counts by sector"));
    methods.append(node("p","Paired median comparisons and the sector CSV cover the three sectors with the most flags. The table above covers all nine sectors.","science-note"));
  }

  function sectorCsv(result) {
    const rows = [["sector","channel","flagged_cells","other_valid_cells","flagged_median_delta_db","other_median_delta_db","difference_of_medians_db","threshold","model_spacing_m","epsg","t0_utc","t1_utc","statistic"]];
    for (const pair of result.paired_sector_deltas || []) for (const i of [0,1]) {
      const f = pair.flagged?.median_delta_hh_hv_db?.[i], b = pair.unflagged?.median_delta_hh_hv_db?.[i];
      rows.push([pair.zone,["HH","HV"][i],pair.flagged?.cells,pair.unflagged?.cells,f,b,finite(f) && finite(b) ? f-b : null,result.threshold,result.roi?.model_spacing_m,result.roi?.epsg,result.acquisitions?.t0?.zeroDopplerStartTime,result.acquisitions?.t1?.zeroDopplerStartTime,"median of paired T1-T0 changes on model grid"]);
    }
    return rows.map(row => row.map(v => `"${String(v ?? "").replaceAll('"','""')}"`).join(",")).join("\r\n");
  }
  function render(result) {
    data = result;
    renderPatterns(); renderScience();
    exportUrls.forEach(url => URL.revokeObjectURL(url)); exportUrls = [];
    for (const [id,contents,type,name] of [["resultJson",JSON.stringify(result,null,2),"application/json","analysis-measurements.json"],["resultCsv",sectorCsv(result),"text/csv;charset=utf-8","sector-measurements.csv"]]) {
      const url = URL.createObjectURL(new Blob([contents],{type})); exportUrls.push(url);
      byId(id).href = url; byId(id).download = name;
    }
  }
  if (typeof document !== "undefined") document.querySelectorAll("[data-evidence-channel]").forEach(b => b.addEventListener("click",() => { channel=Number(b.dataset.evidenceChannel); renderPatterns(); }));
  const api = {render,metrics,sectorCsv};
  if (typeof window !== "undefined") window.NisarEvidence = api;
  if (typeof module !== "undefined") module.exports = api;
})();
