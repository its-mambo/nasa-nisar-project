"use strict";

// Authored guidance from recorded sectors, not validated environmental references.
(() => {
  const zones = ["upper left", "upper center", "upper right", "middle left", "center", "middle right", "lower left", "lower center", "lower right"];
  const titleCase = value => value.replace(/\b\w/g, c => c.toUpperCase());
  const count = value => value.toLocaleString("en-US");
  const signed = value => `${value > 0 ? "+" : ""}${value.toFixed(2)} dB`;
  function sectorBox(zone) {
    const i = zones.indexOf(zone);
    return i < 0 ? null : [(i % 3) / 3, Math.floor(i / 3) / 3, (i % 3 + 1) / 3, (Math.floor(i / 3) + 1) / 3];
  }
  function buildStops(data) {
    // Keep the authored case study out of unrelated live results or future samples.
    if (!data.sample || data.roi?.epsg !== 32640 ||
        JSON.stringify(data.roi?.t0_pixel_box) !== "[5323,2414,11255,6066]" ||
        !data.acquisitions?.t0?.zeroDopplerStartTime?.startsWith("2026-06-21") ||
        !data.acquisitions?.t1?.zeroDopplerStartTime?.startsWith("2026-09-13")) return [];
    const deltas = data.paired_sector_deltas || [];
    const left = deltas.find(row => row.zone === "middle left");
    const right = deltas.find(row => row.zone === "upper right");
    const center = (data.spatial_context?.zones || []).find(row => row.zone === "center");
    if (![left, right].every(row =>
      [row?.flagged, row?.unflagged].every(group => Number.isInteger(group?.cells) && group.cells > 0 &&
        group.median_delta_hh_hv_db?.length === 2 && group.median_delta_hh_hv_db.every(Number.isFinite))) ||
      !Number.isInteger(center?.valid) || center.valid <= 0 || !Number.isInteger(center.flagged) ||
      center.flagged < 0 || center.flagged >= center.valid || center.flagged / center.valid >= .05 ||
      left.flagged.median_delta_hh_hv_db[0] <= 0 || right.flagged.median_delta_hh_hv_db[0] >= 0) return [];
    const paired = row => [
      {label:"HH · flagged cells", value:signed(row.flagged.median_delta_hh_hv_db[0])},
      {label:"HH · other valid cells", value:signed(row.unflagged.median_delta_hh_hv_db[0])},
      {label:"HV · flagged cells", value:signed(row.flagged.median_delta_hh_hv_db[1])}
    ];
    const scope = row => `Medians of paired After − Before differences across ${count(row.flagged.cells)} flagged cells and ${count(row.unflagged.cells)} other valid cells in the entire ${titleCase(row.zone)} sector. These are not measurements of a single patch.`;
    const source = row => `paired_sector_deltas.${deltas.indexOf(row)} · flagged/unflagged.median_delta_hh_hv_db; cells`;
    return [
      {zone:"middle left", title:"A stronger radar return", badge:"Measured signal change",
        intro:"Start where flagged cells have a stronger radar return in the later observation. The detector gives you a place to investigate; the measurements show how the signal changed.",
        look:"Hide Show change and switch between Before and After. Look for differences in brightness and texture. Turn the blue flags back on to see which locations the detector selected.",
        metrics:paired(left), scope:scope(left), source:source(left),
        limit:"Stronger radar return alone does not establish water loss, new vegetation, or any other physical cause.",
        check:"Compare a matching optical image pair at the flagged locations to check whether a shoreline or another visible feature moved."},
      {zone:"center", title:"A mostly unflagged comparison", badge:"Comparison area · unverified",
        intro:"Now inspect a sector where most usable cells were not flagged. A comparison area helps you judge the alerts in context, but it is not a verified unchanged reference.",
        look:"Keep the same display settings and compare Before with After. Show the overlay to distinguish the small flagged share from the rest of this sector.",
        metrics:[{label:"Valid cells flagged", value:`${(100 * center.flagged / center.valid).toFixed(1)}%`},
          {label:"Valid cells without flags", value:count(center.valid - center.flagged)}, {label:"Flagged cells", value:count(center.flagged)}],
        scope:`Counts across all ${count(center.valid)} jointly valid model cells in the entire Center sector. Missing data is excluded.`,
        source:`spatial_context.zones.${data.spatial_context.zones.indexOf(center)} · flagged; valid`,
        limit:"Unflagged does not mean unchanged. The detector can miss changes or score them below the saved threshold.",
        check:"Inspect independent imagery here as well as at flagged locations. This is how you begin looking for missed changes."},
      {zone:"upper right", title:"An unresolved alert", badge:"Environmental meaning unresolved",
        intro:"Here, flagged cells have a weaker radar return in the later observation. This is the opposite signal direction from Stop 1, yet both areas were flagged as possible change.",
        look:"Compare Before and After, then toggle the blue flags. Check whether a visible feature boundary changes or whether only the radar appearance differs.",
        metrics:paired(right), scope:scope(right), source:source(right),
        limit:"These measurements do not tell us whether the cause is a water change, surface conditions, or another effect. This alert has not been established as a true or false environmental detection.",
        check:"Use independent imagery and another radar date to test the explanation. Leave the alert unresolved if the evidence is insufficient."}
    ];
  }
  function create({onSelect, onOverview}) {
    const el = id => document.getElementById(id);
    let stops = [], active = -1;
    function node(tag, text, className) {
      const element = document.createElement(tag);
      if (text !== undefined) element.textContent = text;
      if (className) element.className = className;
      return element;
    }
    function select(index) {
      if (!stops[index]) return;
      active = index;
      const stop = stops[index];
      for (const [i, button] of [...el("investigationStops").children].entries()) {
        button.setAttribute("aria-pressed", String(i === active));
        if (i === active) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      }
      el("investigationDetail").hidden = false;
      el("resultReadingGuide").hidden = true;
      el("investigationStep").textContent = `STOP ${index + 1} OF 3 · ${titleCase(stop.zone)} sector`;
      for (const field of ["Title", "Badge", "Intro", "Look", "Scope", "Limit", "Check"]) {
        el("investigation" + field).textContent = stop[field.toLowerCase()];
      }
      el("investigationMetrics").replaceChildren(...stop.metrics.map(metric => {
        const wrap = node("div"); wrap.append(node("dt", metric.label), node("dd", metric.value)); return wrap;
      }));
      el("investigationSource").textContent = `Saved result fields: ${stop.source}. HH and HV are the two radar measurement channels. This guide uses recorded measurements independently of the experimental Qwen commentary.`;
      el("investigationSourceDetails").open = false;
      el("investigationPrevious").disabled = index === 0;
      el("investigationNext").hidden = index === stops.length - 1;
      el("investigationFinish").hidden = index !== stops.length - 1;
      onSelect({...stop, number:index + 1});
    }
    function overview() {
      active = -1;
      el("investigationDetail").hidden = true;
      el("resultReadingGuide").hidden = false;
      for (const button of el("investigationStops").children) {
        button.setAttribute("aria-pressed", "false"); button.removeAttribute("aria-current");
      }
      onOverview();
    }
    el("investigationPrevious").addEventListener("click", () => select(active - 1));
    el("investigationNext").addEventListener("click", () => select(active + 1));
    el("investigationRecenter").addEventListener("click", () => select(active));
    el("investigationOverview").addEventListener("click", overview);
    el("investigationFinish").addEventListener("click", overview);
    return {render(data) {
      stops = buildStops(data); active = -1;
      el("investigation").hidden = !stops.length;
      el("investigationDetail").hidden = true;
      el("resultReadingGuide").hidden = false;
      el("investigationStops").replaceChildren(...stops.map((stop, i) => {
        const button = node("button", undefined, "investigation-stop"); button.type = "button";
        button.setAttribute("aria-pressed", "false"); button.setAttribute("aria-controls", "investigationDetail");
        button.append(node("span", String(i + 1), "investigation-number"), node("strong", stop.title), node("small", `${titleCase(stop.zone)} sector →`));
        button.addEventListener("click", () => select(i)); return button;
      }));
    }};
  }
  const api = {buildStops, sectorBox, create};
  if (typeof window !== "undefined") window.NisarInvestigation = api;
  if (typeof module !== "undefined") module.exports = api;
})();
