"use strict";

(() => {
  const formatDate = value => new Intl.DateTimeFormat("en-GB", {day:"numeric", month:"short", year:"numeric", timeZone:"UTC"}).format(new Date(value));
  const gapLabel = row => `${Math.abs(row.offset_days)} days ${row.offset_days < 0 ? "before" : "after"} ${row.nearest_radar_label} (${formatDate(row.nearest_radar_date)})`;
  const areaKm2 = hectares => (hectares / 100).toFixed(2);
  const reviewDate = value => new Intl.DateTimeFormat("en-GB", {day:"numeric",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit",timeZone:"Asia/Tashkent",timeZoneName:"short"}).format(new Date(value));
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function link(text, url) {
    const el = node("a",text); el.href = url; el.target = "_blank"; el.rel = "noopener"; return el;
  }
  function render(root, data, {compact=false}={}) {
    const records = data.evidence_report?.external_context || [];
    root.replaceChildren(); root.hidden = !records.length;
    for (const record of records) {
      const head = node("div",undefined,"external-context-head");
      const intro = node("div"); intro.append(node("span","INDEPENDENT SOURCE · REGIONAL CONTEXT","report-eyebrow"),node(compact ? "h4" : "h3",record.title));
      head.append(intro,node("span","Reviewed separately from the model run","external-context-badge"));
      if (!compact) {
        const back=node("button","Back to radar map ↑","results-outline"); back.type="button";
        back.addEventListener("click",()=>document.getElementById("resultMap").scrollIntoView({behavior:"smooth",block:"start"}));
        head.append(back);
      }
      root.append(head,node("p",record.summary,"external-context-summary"));
      const attribution=node("p",undefined,"external-context-small");
      attribution.append(`${record.publisher} · `,link("Read source report ↗",record.sources.at(-1).url)); root.append(attribution);
      const grid = node("div",undefined,"external-context-grid");
      const tableWrap = node("div",undefined,"evidence-table-wrap");
      const table = node("table",undefined,"evidence-table");
      table.append(node("caption","Published Sudochye open-water area · not detector-flagged area"));
      const thead = node("thead"), headers = node("tr");
      for (const text of ["Reference date","Open water","Gap to radar date"]) {const th=node("th",text); th.scope="col"; headers.append(th);}
      thead.append(headers); table.append(thead);
      const tbody = node("tbody");
      for (const row of record.observations) {
        const tr = node("tr"), date=node("th",formatDate(row.date)); date.scope="row";
        const area=node("td",`${areaKm2(row.open_water_ha)} km²`);
        area.append(node("small",`${row.open_water_ha.toLocaleString("en-US")} ha in source`));
        tr.append(date,area,node("td",gapLabel(row))); tbody.append(tr);
      }
      table.append(tbody); tableWrap.append(table,node("p","Source: September report, Table 3, “Sudoche” row, PDF page 5. June value also checked against the June report, Table 2, PDF page 3.","external-context-small"));
      const limits=node("div",undefined,"external-context-limits");
      limits.append(node("h4","What this evidence supports"),node("p","A seasonal water-change question is relevant to this region."),node("h4","What remains unverified"));
      const list=node("ul"); for(const text of record.limitations) list.append(node("li",text));
      limits.append(list,node("strong",record.optical_verification_status),node("p",record.next_check));
      grid.append(tableWrap,limits); root.append(grid);
      if (!compact) {
        const maps=node("details",undefined,"external-context-maps"); maps.dataset.referenceMaps="";
        maps.append(node("summary","View dated Landsat reference maps"),node("p","Find the source label “Sudoche” in the northwest of each map. These are separate regional reference views with different extents, not a registered before-and-after overlay. Blue marks the publisher’s water classification.","external-context-small"));
        const figures=node("div",undefined,"external-context-figures");
        for (const source of record.sources) {
          const figure=node("figure"), anchor=link("",source.image_url), image=node("img");
          image.src=source.image_url; image.loading="lazy";
          image.alt=`${source.sensor} regional reference map dated ${formatDate(source.observation_date)}; Sudochye is labeled Sudoche; blue marks the source water classification`;
          anchor.append(image);
          const caption=node("figcaption"); caption.append(node("strong",formatDate(source.observation_date)),node("span",gapLabel(source)),node("small",`${source.sensor} · ${source.figure}`),link("Open full-size reference map ↗",source.image_url));
          figure.append(anchor,caption); figures.append(figure);
        }
        maps.append(figures); root.append(maps);
      }
      const provenance=node("details",undefined,"external-context-provenance");
      provenance.append(node("summary","Sources, geographic scope, and provenance"),node("p",record.publisher),node("p",record.geographic_scope),node("p",`Reviewed source snapshot: ${reviewDate(record.retrieved_at)}. Values transcribed from the cited tables; hectares ÷ 100 gives km². No live web lookup is performed during playback.`));
      const sources=node("div",undefined,"external-context-sources");
      for (const source of record.sources) {
        const item=node("div"); item.append(node("strong",source.title),link("Official source PDF ↗",source.url),node("small",`Reviewed source PDF SHA-256: ${source.sha256}`)); sources.append(item);
      }
      provenance.append(sources); root.append(provenance);
    }
  }
  const api={render,areaKm2,gapLabel};
  if (typeof window!=="undefined") window.NisarExternalContext=api;
  if (typeof module!=="undefined") module.exports=api;
})();
