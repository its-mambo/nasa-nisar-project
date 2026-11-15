"""Build a report from measured radar evidence and detector output."""
import math
import re

SCHEMA_VERSION = "nisar_evidence_report_v2"

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def evidence_catalog(result):
    """Only application-calculated values become measured/detector evidence."""
    roi = result.get("roi") or {}
    flags, valid = result.get("changed_pixels"), result.get("valid_pixels")
    spacing = roi.get("model_spacing_m")
    evidence = []
    if finite(flags) and finite(valid) and valid > 0 and finite(spacing) and spacing > 0:
        area, pct = flags*spacing**2/1e6,100*flags/valid
        evidence.append(dict(id="detector:summary",kind="detector",label="Detector-flagged area",
            display=f"{area:,.2f} km² · {pct:.1f}% of valid model cells",value=area,unit="km²",
            cells=flags,valid_cells=valid,threshold=result.get("threshold"),
            source_paths=["changed_pixels","valid_pixels","roi.model_spacing_m","threshold"]))
    width,height = roi.get("model_grid_width_px"),roi.get("model_grid_height_px")
    if finite(valid) and finite(width) and finite(height) and width*height > 0:
        evidence.append(dict(id="coverage:paired",kind="measured",label="Usable model coverage",
            display=f"{100*valid/(width*height):.1f}% · {valid:,} jointly valid cells",value=100*valid/(width*height),unit="%",
            source_paths=["valid_pixels","roi.model_grid_width_px","roi.model_grid_height_px"]))
    for i,row in enumerate(result.get("paired_sector_deltas") or []):
        zone = row.get("zone","")
        slug = re.sub(r"[^a-z0-9]+","_",zone.lower()).strip("_")
        for group,name in (("flagged","flagged"),("unflagged","other valid cells")):
            values = (row.get(group) or {}).get("median_delta_hh_hv_db") or []
            for channel,index in (("HH",0),("HV",1)):
                value = values[index] if len(values)>index else None
                if finite(value):
                    evidence.append(dict(id=f"sector:{slug}:{channel.lower()}:{group}",kind="measured",
                        label=f"{zone.title()} · {channel} · {name}",display=f"{value:+.2f} dB",value=value,unit="dB",
                        cells=(row.get(group) or {}).get("cells"),sector=zone,channel=channel,group=group,
                        source_paths=[f"paired_sector_deltas.{i}.{group}.median_delta_hh_hv_db.{index}"],
                        method="Median of paired T1-minus-T0 model-cell backscatter differences"))
    for date,name in (("t0","Before"),("t1","After")):
        stamp = (result.get("acquisitions",{}).get(date) or {}).get("zeroDopplerStartTime")
        evidence.append(dict(id=f"image:{date}",kind="image",label=f"{name} image ({date.upper()})",
                             display=stamp or "Acquisition time not recorded",source_paths=[f'images["{date}.png"]']))
    return evidence


def build_report(result):
    evidence = evidence_catalog(result)
    summary = next((item for item in evidence if item["id"]=="detector:summary"),None)
    flags = result.get("changed_pixels")
    finding = ("There is not enough jointly valid model data to summarize detector flags." if not finite(result.get("valid_pixels")) or result["valid_pixels"]<=0 else
               "No model cells were flagged at the selected threshold." if flags == 0 else
               f"The detector flagged {summary['display'].lower()}." if summary else
               "There is not enough jointly valid model data to summarize detector flags.")
    warning = bool(result.get("warnings"))
    checks = [dict(source="Optical imagery or another radar acquisition",
                   compare=("Compare the same flagged locations near both dates." if flags else
                            "Check the same location with independent imagery if a change is suspected."),
                   supports="A corresponding signal or surface change supports further investigation.",
                   weakens="No matching change weakens the case for a persistent surface change.")]
    paired = result.get("paired_sector_deltas") or []
    lead = next((row for row in paired if any(finite(v) for v in (row.get("flagged",{}).get("median_delta_hh_hv_db") or []))),None)
    featured = [item["id"] for item in evidence if item["kind"]=="measured" and not item.get("sector")]
    if lead:
        featured.extend(item["id"] for item in evidence if item.get("sector")==lead.get("zone") and item.get("group")=="flagged")
    return dict(schema_version=SCHEMA_VERSION,finding=dict(kind="detector",text=finding,evidence_refs=["detector:summary"] if summary else []),
        evidence=evidence,featured_evidence=featured[:4],checks=checks,
        caution="Resolve acquisition comparability warnings before physical attribution." if warning else
                "Radar-signal change and detector flags do not confirm an environmental cause.",
        validation="Detector flags are not independently validated ground truth. Suggested checks have not been performed.",
        external_context=[])


def detector_result(result):
    """Project a legacy saved result onto the detector-only report contract."""
    result = dict(result)
    for key in ("interpretation", "decoder_inputs_url", "decoder_profile", "pattern_evidence",
                "predictions", "candidate_regions", "candidate_crops", "representatives",
                "reason", "score_note", "min_region_pixels", "min_valid_fraction", "status"):
        result.pop(key, None)
    timing = result.get("timing_s")
    if isinstance(timing, dict):
        result["timing_s"] = {key: timing[key] for key in ("input_and_alignment", "detection",
                                "measurements", "total") if key in timing}
    result["report_sections"] = [section for section in result.get("report_sections", [])
                                 if section.get("title") != "Model label and interpretation"]
    result["evidence_report"] = build_report(result)
    return result


def render_export(result,format="md"):
    """Export measured evidence and detector results."""
    if format not in {"md","txt"}:
        raise ValueError("Choose Markdown or plain text")
    report = result.get("evidence_report") or build_report(result)
    if report.get("schema_version") != SCHEMA_VERSION:
        report = {**build_report(result), "external_context": report.get("external_context", [])}
    md = format=="md"
    def escape(text):
        text = str(text)
        return re.sub(r"([\\`*_{}\[\]<>#|])",r"\\\1",text) if md else text
    lines = []
    def heading(text,level=2):
        lines.extend([f"{'#'*level} {text}" if md else text.upper(),""])
    def paragraph(text):
        lines.extend([escape(text),""])
    heading("Change detection evidence",1)
    acq = result.get("acquisitions") or {}
    paragraph("Before (T0): "+str((acq.get("t0") or {}).get("zeroDopplerStartTime","Not recorded")))
    paragraph("After (T1): "+str((acq.get("t1") or {}).get("zeroDopplerStartTime","Not recorded")))
    heading("Main finding · detector output")
    paragraph(report["finding"]["text"])
    heading("Evidence · application-calculated values")
    paragraph("HH/HV sector values are medians of paired After-minus-Before backscatter changes on the model grid. Negative dB means a weaker return; positive dB means a stronger return.")
    for item in report["evidence"]:
        if item["kind"] == "image":
            continue
        kind = "Detector output" if item["kind"]=="detector" else "Measured"
        samples = f"; {item['cells']:,} cells" if isinstance(item.get("cells"),int) else ""
        lines.append(f"- [{kind}] {escape(item['label'])}: {escape(item['display'])}{samples} ({escape(item['id'])})")
    lines.append("")
    roi = result.get("roi") or {}
    grid = roi.get("model_grid_width_px",0)*roi.get("model_grid_height_px",0)
    valid,flags = result.get("valid_pixels"),result.get("changed_pixels")
    if grid>0 and finite(valid) and finite(flags):
        paragraph(f"Model-grid composition: {flags:,} flagged; {valid-flags:,} valid without flags; {grid-valid:,} without paired model data. Unflagged is not confirmed stable.")
    for context in report.get("external_context") or []:
        heading(context["title"] + " · external regional context")
        paragraph(context["summary"])
        paragraph("Geographic scope: " + context["geographic_scope"])
        paragraph("Context record: " + context["id"] + "; sources reviewed separately: " + context["retrieved_at"])
        for row in context["observations"]:
            gap = f"{abs(row['offset_days'])} days {'before' if row['offset_days'] < 0 else 'after'} {row['nearest_radar_label']} radar ({row['nearest_radar_date']})"
            paragraph(f"{row['date']}: {row['open_water_ha']:,} ha = {row['open_water_ha']/100:.2f} km² of published open water; {gap}. Source: {row['source_id']}; {row['source_locator']}.")
        for limitation in context["limitations"]:
            paragraph("Limitation: " + limitation)
        paragraph(context["optical_verification_status"] + ". " + context["next_check"])
        for source in context["sources"]:
            paragraph(context["publisher"] + ": " + source["title"])
            paragraph("Official source: " + source["url"])
            paragraph(f"Reference image: {source['sensor']}; observation {source['observation_date']}; {source['figure']}.")
            paragraph("Source PDF SHA-256: " + source["sha256"] + "; extracted reference image SHA-256: " + source["image_sha256"])
    paragraph(report["caution"])
    heading("Next checks · recommended sources, not consulted")
    for check in report["checks"]:
        paragraph(check["source"]+": "+check["compare"])
        paragraph("Would support: "+check["supports"])
        paragraph("Would weaken: "+check["weakens"])
    heading("Scope & reproducibility")
    sample = result.get("sample")
    if sample:
        paragraph("Saved analysis sample: generated once and replayed without running models. Inputs, ROI, and threshold are fixed.")
        paragraph("Sample revision: "+str(sample["revision"])+"; generated: "+str(sample["generated_at"]))
        for date, source in sample["sources"].items():
            paragraph(f"{date.upper()} source: {source['path']}; SHA-256: {source['sha256']}")
    for label,value in (("Threshold",result.get("threshold")),("Map reference",f"EPSG:{roi.get('epsg','Not recorded')}"),
                        ("Model cell spacing (m)",roi.get("model_spacing_m")),("T0 pixel box",roi.get("t0_pixel_box")),
                        ("Detector checkpoint SHA-256",(result.get("analysis_settings") or {}).get("detector_sha256"))):
        lines.append(f"- {label}: {escape(value if value is not None else 'Not recorded')}")
    lines.append("")
    for warning in result.get("warnings") or []:
        paragraph("Acquisition warning: "+warning)
    paragraph(report["validation"])
    paragraph("Regional source snapshots were reviewed separately from the original model run. They do not validate individual flags or environmental causes."
              if report.get("external_context") else
              "No external web sources were retrieved for this report. Charts in the app use the counts and dB values listed above.")
    return "\n".join(lines).rstrip()+"\n"
