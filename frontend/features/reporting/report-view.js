"use strict";

(() => {
  const finite = n => typeof n === "number" && Number.isFinite(n);
  const fmt = n => finite(n) ? n.toLocaleString("en-US") : "Not recorded";
  const label = kind => ({measured:"Measured",detector:"Detector output",image:"Image reference",ai:"AI observation · unverified"})[kind] || kind;
  function node(tag,text,className) {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = String(text);
    if (className) n.className = className;
    return n;
  }
  function composition(data) {
    const roi = data.roi || {}, total = roi.model_grid_width_px*roi.model_grid_height_px;
    const valid = data.valid_pixels, flagged = data.changed_pixels;
    if (![total,valid,flagged].every(finite) || total<=0 || flagged<0 || flagged>valid || valid>total) return null;
    return {total,valid,parts:[{label:"Flagged",count:flagged,color:"#2576d5"},
      {label:"Valid, without flags",count:valid-flagged,color:"#91aebc"},
      {label:"No paired model data",count:total-valid,color:"#e4e9ee"}]};
  }
  function sourceChip(item,compact=false) {
    const button = node("button",undefined,`report-source-chip${compact ? " compact" : ""}`); button.type = "button";
    const name=compact && item.kind==="ai" ? `Observation ${item.id.split(":").at(-1)}` : item.label;
    button.append(node("span",label(item.kind),`claim-badge ${item.kind}`),node("strong",name));
    if(!compact) button.append(node("small",item.display));
    const detail = node("div",undefined,"report-source-detail"); detail.hidden = true;
    button.setAttribute("aria-expanded","false");
    if(compact) detail.append(node("p",item.kind==="ai" ? item.label : item.display));
    detail.append(node("p",item.method || (item.kind === "image" ? "Qwen received this image; descriptions of its features remain unverified AI observations." : item.kind === "ai" ? "This is an unverified model observation, not independently checked evidence." : "Calculated by the application from the recorded analysis.")));
    if (item.cells != null) detail.append(node("p",`${fmt(item.cells)} cells`));
    if (item.source_paths) detail.append(node("p",`Data fields: ${item.source_paths.join(", ")}`));
    button.addEventListener("click",() => { detail.hidden=!detail.hidden; button.setAttribute("aria-expanded",String(!detail.hidden)); });
    const wrap = node("div",undefined,"report-source-wrap"); wrap.append(button,detail); return wrap;
  }
  function references(ids,catalog) {
    const root = node("div",undefined,"report-claim-refs");
    for (const id of ids || []) {
      const item = catalog.get(id);
      if (item) root.append(sourceChip(item,true));
    }
    return root;
  }
  function coverageChart(data) {
    const figure = node("figure",undefined,"report-figure"); figure.append(node("figcaption","Model-grid coverage & detector flags"));
    const values = composition(data);
    if (!values) { figure.append(node("p","Coverage counts are unavailable for this result.")); return figure; }
    const [flagged,valid,missing] = values.parts;
    const donut = node("div",undefined,"report-donut");
    const a = flagged.count/values.total*360, b = (flagged.count+valid.count)/values.total*360;
    donut.style.background = `conic-gradient(${flagged.color} 0deg ${a}deg, ${valid.color} ${a}deg ${b}deg, ${missing.color} ${b}deg 360deg)`;
    donut.setAttribute("role","img");
    donut.setAttribute("aria-label",values.parts.map(p=>`${p.label}: ${fmt(p.count)} model cells`).join("; "));
    const center = node("div",undefined,"report-donut-center"); center.append(node("strong",`${(100*values.valid/values.total).toFixed(1)}%`),node("span","usable data")); donut.append(center);
    const legend = node("ul",undefined,"report-donut-legend");
    for (const part of values.parts) {
      const row=node("li"), swatch=node("i"); swatch.style.background=part.color; swatch.setAttribute("aria-hidden","true");
      const text=node("span",part.label); text.append(node("small",`${fmt(part.count)} cells · ${(100*part.count/values.total).toFixed(1)}% of all model cells`)); row.append(swatch,text); legend.append(row);
    }
    const content = node("div",undefined,"report-donut-layout"); content.append(donut,legend); figure.append(content,node("p","Unflagged cells are not confirmed stable. Missing pairs are excluded from the detection percentage.","report-small"));
    return figure;
  }
  function sectorChart(data,onInspectSector) {
    const figure=node("figure",undefined,"report-figure");
    const row=(data.paired_sector_deltas || []).find(p => (p.flagged?.median_delta_hh_hv_db || []).some(finite));
    figure.append(node("figcaption",row ? `${row.zone.replace(/\b\w/g,c=>c.toUpperCase())} · paired signal change` : "Paired signal change"));
    if (!row) { figure.append(node("p","There are too few cells for a paired sector comparison.","report-small")); return figure; }
    const values=[...(row.flagged?.median_delta_hh_hv_db || []),...(row.unflagged?.median_delta_hh_hv_db || [])].filter(finite);
    const scale=Math.max(1,...values.map(Math.abs));
    for (const i of [0,1]) for (const [key,name] of [["flagged","Flagged"],["unflagged","Other valid"]]) {
      const value=row[key]?.median_delta_hh_hv_db?.[i];
      const line=node("div",undefined,"report-signal-row"), track=node("div",undefined,"signal-track"); track.setAttribute("aria-hidden","true");
      const bar=node("span",undefined,key==="flagged" ? "flagged" : "background");
      if (finite(value)) { bar.style.left=`${50+Math.min(0,value/scale*48)}%`; bar.style.width=`${Math.abs(value)/scale*48}%`; }
      track.append(bar); line.append(node("span",`${["HH","HV"][i]} · ${name}`),track,node("strong",finite(value) ? `${value>0?"+":""}${value.toFixed(2)} dB` : "n/a")); figure.append(line);
    }
    figure.append(node("p","← Weaker return · 0 · Stronger return →\nMedian paired After − Before differences; n/a means fewer than 32 cells.","report-small"));
    if (onInspectSector) {
      const inspect=node("button","Inspect this sector on the map ↗","report-map-link"); inspect.type="button";
      inspect.addEventListener("click",()=>onInspectSector(row.zone)); figure.append(inspect);
    }
    return figure;
  }
  function render(data,options={}) {
    const report=data.evidence_report;
    if (!report) return;
    const byId=id=>document.getElementById(id);
    const catalog=new Map(report.evidence.map(item=>[item.id,item]));
    for (const o of report.ai.observations) catalog.set(o.id,{id:o.id,kind:"ai",label:o.text,display:"Unverified interpretation of the supplied images/evidence"});
    byId("reportFindingTitle").textContent=report.finding.text;
    byId("reportFindingNote").textContent=`Detector output · threshold ${data.threshold ?? "not recorded"}. ${report.caution}`;
    byId("reportEvidence").replaceChildren(...report.featured_evidence.map(id=>catalog.get(id)).filter(Boolean).map(item=>sourceChip(item)));
    byId("reportCharts").replaceChildren(coverageChart(data),sectorChart(data,options.onInspectSector));
    const interpretations=byId("reportInterpretation"); interpretations.replaceChildren();
    const fallback = report.ai.status === "text_fallback" && typeof report.ai.text === "string";
    const observations=node("section",undefined,`report-claim-card${fallback ? " report-text-fallback" : ""}`);
    observations.append(node("span",fallback ? "Qwen explanation · unverified" : "AI observations · unverified","claim-badge ai"));
    if (fallback) {
      observations.append(node("h5","What the AI suggests"),node("p",report.ai.presentation_note,"report-small"));
      // Model output remains text: no generated HTML, links, or validated source chips.
      observations.append(node("p",report.ai.text,"report-model-text"));
    } else if (!report.ai.observations.length) observations.append(node("p",report.ai.status==="structured" ? "No specific visual observation was proposed." : "A usable structured AI interpretation is unavailable for this run. You can still inspect the measured evidence and suggested checks."));
    for (const observation of report.ai.observations) {
      observations.append(node("p",observation.text),references(observation.evidence_refs,catalog));
    }
    const possible=node("section",undefined,"report-claim-card hypothesis"); possible.append(node("span",report.attribution_status,"claim-badge hypothesis"));
    const hypothesis=report.ai.hypothesis;
    if (hypothesis) {
      possible.append(node("h5",hypothesis.title),node("p",hypothesis.explanation),node("h6","Supporting references"),references(hypothesis.evidence_refs,catalog),node("h6","Still unresolved"));
      const list=node("ul"); for(const uncertainty of hypothesis.uncertainties) list.append(node("li",uncertainty)); possible.append(list);
    } else possible.append(node("h5","Cause unresolved"),node("p","The available evidence does not establish a specific physical cause. Use an independent check before attributing the change."));
    interpretations.append(observations);
    if (!fallback) interpretations.append(possible);
    const checks=byId("reportChecks"); checks.replaceChildren();
    for(const check of report.ai.verification) {
      const card=node("section",undefined,"report-check-card"); card.append(node("h5",check.source),node("p",check.compare));
      const criteria=node("dl"); for(const [name,value] of [["Would support",check.supports],["Would weaken",check.weakens]]) {const row=node("div"); row.append(node("dt",name),node("dd",value)); criteria.append(row);}
      card.append(criteria); checks.append(card);
    }
    const refRoot=byId("reportReferences"); refRoot.replaceChildren();
    for(const item of report.evidence) refRoot.append(sourceChip(item));
    byId("reportValidation").textContent=report.validation+(report.ai.format_note ? ` Format note: ${report.ai.format_note}` : "");
    const rev=new URL(data.images?.["t0.png"] || "/",location.origin).searchParams.get("v");
    for(const [id,ext] of [["reportMarkdown","md"],["reportText","txt"]]) {
      byId(id).hidden=!Nisar.isResultRevision(rev);
      if (rev) { byId(id).href=`/report.${ext}?v=${encodeURIComponent(rev)}`; byId(id).download=`nisar-evidence-report-${rev}.${ext}`; }
    }
  }
  const api={render,composition};
  if(typeof window!=="undefined") window.NisarReport=api;
  if(typeof module!=="undefined") module.exports=api;
})();
