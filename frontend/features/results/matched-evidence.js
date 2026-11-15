"use strict";

(() => {
  const percent = value => `${(100 * value).toFixed(1)}%`;
  const signed = value => `${value >= 0 ? "+" : ""}${value.toFixed(2)} dB`;
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function link(text, url, download=false) {
    const el = node("a", text); el.href = url;
    if (download) el.download = "";
    else { el.target = "_blank"; el.rel = "noopener"; }
    return el;
  }
  function figure(url, caption, alt, point) {
    const el = node("figure"), image = node("img");
    image.src = url; image.alt = alt; image.loading = "lazy";
    const frame = node("div", undefined, "matched-image-frame"); frame.append(image);
    if (point) {
      const marker = node("span", undefined, "matched-crosshair"); marker.setAttribute("aria-hidden", "true");
      marker.style.left = `${point[0] * 100}%`; marker.style.top = `${point[1] * 100}%`; frame.append(marker);
    }
    el.append(frame, node("figcaption", caption)); return el;
  }
  function render(root, data, {onInspect, compact=false}={}) {
    const pilot = data.evidence_report?.matched_evidence;
    root.replaceChildren(); root.hidden = !pilot;
    if (!pilot) return;
    const head = node("div", undefined, "external-context-head");
    const title = node("div"); title.append(node("span", "LOCATION-MATCHED EVIDENCE · EXPLORATORY PILOT", "report-eyebrow"), node("h3", pilot.title));
    head.append(title, node("span", "4 selected locations · accuracy not established", "external-context-badge"));
    root.append(head, node("p", pilot.summary), node("p", pilot.scope, "matched-caution"));
    if (compact) {
      const list = node("ul");
      for (const site of pilot.sites) list.append(node("li", `${site.title}: ${site.conclusion}`));
      root.append(list, node("p", pilot.review_status, "external-context-small"));
      return;
    }
    root.append(node("p", "Landsat 8 · 18 Jun 2026 (3 days before Before radar) → Landsat 9 · 14 Sep 2026 (1 day after After radar). Each pair covers the same 3 km square. Grey cells are excluded by source quality checks.", "matched-dates"));
    const tabs = node("div", undefined, "matched-tabs"); tabs.setAttribute("role", "group"); tabs.setAttribute("aria-label", "Location-matched optical checks");
    const detail = node("div", undefined, "matched-detail");
    let selected = 0, mode = "natural";
    const buttons = pilot.sites.map((site, index) => {
      const button = node("button", `${index + 1} · ${site.title}`); button.type = "button";
      button.addEventListener("click", () => { selected = index; show(); });
      tabs.append(button); return button;
    });
    function show() {
      const site = pilot.sites[selected];
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === selected)));
      detail.replaceChildren();
      const heading = node("div", undefined, "matched-site-head");
      heading.append(node("h4", site.title), node("span", site.badge, "investigation-badge"));
      if (onInspect) {
        const inspect = node("button", "Locate this square on the radar map ↑", "results-outline"); inspect.type = "button";
        inspect.addEventListener("click", () => onInspect(site)); heading.append(inspect);
      }
      detail.append(heading, node("p", site.observation));
      const controls = node("div", undefined, "matched-controls"); controls.setAttribute("role", "group"); controls.setAttribute("aria-label", "Optical colour display");
      for (const [value, label] of [["natural", "Natural colour"], ["false_color", "SWIR / NIR / red"]]) {
        const button = node("button", label); button.type = "button"; button.setAttribute("aria-pressed", String(mode === value));
        button.addEventListener("click", () => { mode = value; show(); }); controls.append(button);
      }
      controls.append(node("span", "Display only · measurements and radar controls stay unchanged")); detail.append(controls);
      const pair = node("div", undefined, "matched-pair");
      for (const [label, date] of [["before", "18 June 2026"], ["after", "14 September 2026"]]) {
        pair.append(figure(site.images[`${label}_${mode}`], `${date} · usable ${percent(site.optical[label].usable_fraction)} of native optical cells`, `${site.title}, ${label} optical image, same mapped 3 km square; grey is excluded reference data`, site.comparison_point?.image_fraction));
      }
      detail.append(pair, node("p", "North is up. Both dates use the same colour stretch. Optical detail is 30 m; displaying it on the radar grid adds no detail.", "external-context-small"));
      if (site.comparison_point) detail.append(node("p", `The crosshair marks the same mapped point at both dates: ${site.comparison_point.easting_m} E, ${site.comparison_point.northing_m} N (EPSG:32640). Here the source water flag changes from present to absent; this is supporting algorithmic evidence, not ground truth.`, "external-context-small"));
      detail.append(node("p", site.conclusion, "matched-conclusion"), node("p", `Next check: ${site.next_check}`));
      const measurements = node("dl", undefined, "matched-metrics");
      for (const [label, value] of [
        ["Local paired HH / HV change", site.radar.median_delta_hh_hv_db.map(signed).join(" / ")],
        ["Visible saved flags", `${percent(site.radar.visible_flag_fraction)} of paired radar cells · lower bound`],
        ["Reference quality · Before / After", `${percent(site.optical.before.usable_fraction)} / ${percent(site.optical.after.usable_fraction)} usable`],
      ]) { const row = node("div"); row.append(node("dt", label), node("dd", value)); measurements.append(row); }
      detail.append(measurements, node("p", "HH/HV values are medians of paired After − Before differences across all valid radar cells in this square, not just flagged cells. Visible flags are not the complete binary detector mask.", "external-context-small"));
      const technical = node("details"); technical.append(node("summary", "Exact location, radar crops, measurements & source files"));
      technical.append(node("p", `Centre: ${site.center_lon_lat[1].toFixed(6)}°N, ${site.center_lon_lat[0].toFixed(6)}°E. EPSG:${pilot.epsg} west/south/east/north: ${site.bounds_wsen.join(", ")} m. Area: ${site.area_km2} km².`));
      const radar = node("div", undefined, "matched-radar");
      for (const [key, label] of [["radar_before", "Radar · 21 June"], ["radar_after", "Radar · 13 September"], ["overlay", "Visible saved detector flags"]]) radar.append(figure(site.images[key], label, `${site.title}: ${label}, exact same square`, site.comparison_point?.image_fraction));
      technical.append(radar);
      const table = node("table", undefined, "evidence-table"), body = node("tbody");
      table.append(node("caption", "Exploratory optical diagnostics · algorithms, not reference truth"));
      const header = node("tr"); for (const text of ["Diagnostic", "18 June", "14 September"]) header.append(node("th", text)); body.append(header);
      for (const [label, key, format] of [["Median water index (MNDWI)", "median_mndwi", x => x == null ? "Unavailable" : x.toFixed(3)], ["Index above 0.15", "water_like_fraction", x => x == null ? "Unavailable" : percent(x)], ["Source QA water flag", "source_water_flag_fraction", x => x == null ? "Unavailable" : percent(x)]]) {
        const row = node("tr"); row.append(node("th", label), ...["before", "after"].map(date => node("td", format(site.optical[date][key])))); body.append(row);
      }
      table.append(body); technical.append(table, node("p", pilot.methods.water_like_note));
      const downloads = node("div", undefined, "matched-downloads");
      downloads.append(link("↓ Before source GeoTIFF (7 bands)", site.source_crops.before, true), link("↓ After source GeoTIFF (7 bands)", site.source_crops.after, true));
      technical.append(downloads); detail.append(technical);
    }
    root.append(tabs, detail); show();
    const methods = node("details", undefined, "matched-methods"); methods.append(node("summary", "How these checks were made, sources & limitations"));
    methods.append(node("p", pilot.review_status), node("p", pilot.selection));
    for (const key of ["quality", "display", "statistics"]) methods.append(node("p", pilot.methods[key]));
    const limits = node("ul"); pilot.limitations.forEach(text => limits.append(node("li", text))); methods.append(limits);
    for (const source of pilot.sources) methods.append(link(`${source.sensor} · ${source.id} · source metadata ↗`, source.url));
    pilot.method_sources.forEach((url, index) => methods.append(link(["USGS quality flags ↗", "USGS reflectance scaling ↗", "Reference data access ↗"][index], url)));
    methods.append(link("↓ Site summary CSV", "/matched-source/sites.csv", true));
    methods.append(link("↓ Complete pilot measurements, methods & checksums JSON", "/matched-source/manifest.json", true)); root.append(methods);
  }
  const api = {render};
  if (typeof window !== "undefined") window.NisarMatchedEvidence = api;
  if (typeof module !== "undefined") module.exports = api;
})();
