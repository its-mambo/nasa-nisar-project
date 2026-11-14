"""A report contract separating computed evidence from unverified AI claims."""
import json
import math
import re

SCHEMA_VERSION = "nisar_evidence_report_v1"
SOURCE_TYPES = {
    "optical_imagery":"Optical imagery",
    "additional_radar_date":"Another radar acquisition",
    "water_records":"Water-level or water-extent records",
    "land_cover_reference":"Land-cover reference data",
    "field_observations":"Field observations",
    "weather_records":"Weather records",
}
MEASUREMENT = re.compile(r"[+−-]?\d+(?:\.\d+)?\s*(?:dB|%|km[²2]?|m[²2]?|cells|pixels)(?!\w)",re.I)


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


def _text(value,limit=350):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:
        raise ValueError("Missing or overlong interpretation text")
    value = " ".join(value.split())
    if MEASUREMENT.search(value):
        raise ValueError("AI text repeated a numeric measurement instead of referencing evidence")
    if re.search(r"https?://|www\.",value,re.I):
        raise ValueError("AI text invented a URL without a retrieved source record")
    return value


def _keys(value,expected):
    if not isinstance(value,dict) or set(value) != set(expected):
        raise ValueError("Unexpected interpretation fields")


def _refs(value,known):
    if (not isinstance(value,list) or not 1<=len(value)<=4
            or any(not isinstance(ref,str) or ref not in known for ref in value)):
        raise ValueError("Missing or unknown evidence references")
    return list(dict.fromkeys(value))


def parse_interpretation(raw,catalog):
    """Check format/references only; this does not validate physical attribution."""
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*","",raw)[:-3].strip()
    parsed = json.loads(raw)
    _keys(parsed,["observations","hypothesis","verification"])
    known = {item["id"] for item in catalog}
    observations = parsed["observations"]
    if not isinstance(observations,list) or len(observations)>2:
        raise ValueError("Expected at most two AI observations")
    normalized, seen = [], set()
    for i,item in enumerate(observations,1):
        _keys(item,["text","evidence_refs"])
        text = _text(item["text"],280)
        if text.casefold() in seen:
            raise ValueError("Repeated observation")
        seen.add(text.casefold())
        normalized.append(dict(id=f"ai:observation:{i}",text=text,evidence_refs=_refs(item["evidence_refs"],known)))
    hypothesis = parsed["hypothesis"]
    if hypothesis is not None:
        _keys(hypothesis,["title","explanation","evidence_refs","uncertainties"])
        uncertainties = hypothesis["uncertainties"]
        if not isinstance(uncertainties,list) or not 1<=len(uncertainties)<=2:
            raise ValueError("A hypothesis must state its unresolved assumptions")
        hypothesis = dict(title=_text(hypothesis["title"],100),explanation=_text(hypothesis["explanation"],350),
            evidence_refs=_refs(hypothesis["evidence_refs"],known|{o["id"] for o in normalized}),
            uncertainties=[_text(s,180) for s in uncertainties])
        if hypothesis["explanation"].casefold() in seen:
            raise ValueError("Hypothesis repeated an observation")
    checks = parsed["verification"]
    if not isinstance(checks,list) or not 1<=len(checks)<=2:
        raise ValueError("Expected one or two independent verification steps")
    verification = []
    for item in checks:
        _keys(item,["source_type","compare","supports","weakens"])
        if item["source_type"] not in SOURCE_TYPES:
            raise ValueError("Unknown independent-source category")
        verification.append(dict(source_type=item["source_type"],source=SOURCE_TYPES[item["source_type"]],
            compare=_text(item["compare"],260),supports=_text(item["supports"],200),weakens=_text(item["weakens"],200),
            consulted=False))
    return dict(observations=normalized,hypothesis=hypothesis,verification=verification)


def build_report(result):
    evidence = evidence_catalog(result)
    summary = next((item for item in evidence if item["id"]=="detector:summary"),None)
    flags = result.get("changed_pixels")
    finding = ("There is not enough jointly valid model data to summarize detector flags." if not finite(result.get("valid_pixels")) or result["valid_pixels"]<=0 else
               "No model cells were flagged at the selected threshold." if flags == 0 else
               f"The detector flagged {summary['display'].lower()}." if summary else
               "There is not enough jointly valid model data to summarize detector flags.")
    ai = dict(status="unavailable",observations=[],hypothesis=None,verification=[])
    raw = (result.get("interpretation") or {}).get("summary")
    if isinstance(raw,str) and raw.strip():
        try:
            ai.update(parse_interpretation(raw,evidence),status="structured")
        except (ValueError,TypeError,KeyError) as exc:
            ai["format_note"] = str(exc)
    warning = bool(result.get("warnings"))
    if warning or not finite(flags) or flags <= 0:
        ai["hypothesis"] = None
    if not ai["verification"]:
        ai["verification"] = [dict(source_type="optical_imagery",source=SOURCE_TYPES["optical_imagery"],consulted=False,
            compare="Inspect spatially matched optical images near both acquisition dates at the flagged locations.",
            supports="A corresponding change in boundaries or surface appearance supports investigating a physical change.",
            weakens="Unchanged features weaken a feature-specific explanation; consider broader radar-acquisition effects.")]
    paired = result.get("paired_sector_deltas") or []
    lead = next((row for row in paired if any(finite(v) for v in (row.get("flagged",{}).get("median_delta_hh_hv_db") or []))),None)
    featured = [item["id"] for item in evidence if item["kind"]=="measured" and not item.get("sector")]
    if lead:
        featured.extend(item["id"] for item in evidence if item.get("sector")==lead.get("zone") and item.get("group")=="flagged")
    return dict(schema_version=SCHEMA_VERSION,finding=dict(kind="detector",text=finding,evidence_refs=["detector:summary"] if summary else []),
        evidence=evidence,featured_evidence=featured[:4],ai=ai,
        attribution_status="Cause unresolved" if ai["hypothesis"] is None else "Unconfirmed hypothesis",
        caution="Resolve acquisition comparability warnings before physical attribution." if warning else
                "Radar-signal change and detector flags do not confirm an environmental cause.",
        validation="AI output was checked for format and valid references, not independently validated scientifically.",
        external_context=[],
        provenance=dict(decoder=(result.get("interpretation") or {}).get("decoder"),threshold=result.get("threshold"),
                        epsg=(result.get("roi") or {}).get("epsg")))


def render_export(result,format="md"):
    """Portable text exports contain the same classified claims and exact values."""
    if format not in {"md","txt"}:
        raise ValueError("Choose Markdown or plain text")
    report = result.get("evidence_report") or build_report(result)
    md = format=="md"
    def escape(text):
        text = str(text)
        return re.sub(r"([\\`*_{}\[\]<>#|])",r"\\\1",text) if md else text
    lines = []
    def heading(text,level=2):
        lines.extend([f"{'#'*level} {text}" if md else text.upper(),""])
    def paragraph(text):
        lines.extend([escape(text),""])
    heading("Evidence & interpretation",1)
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
    heading("AI observations · unverified")
    if report["ai"]["observations"]:
        for observation in report["ai"]["observations"]:
            paragraph(observation["text"]+" Sources: "+", ".join(observation["evidence_refs"]))
    else:
        paragraph("A usable structured AI interpretation is unavailable. The measured evidence remains available above.")
    heading("Possible explanation · "+report["attribution_status"])
    hypothesis = report["ai"]["hypothesis"]
    if hypothesis:
        paragraph(hypothesis["title"]+": "+hypothesis["explanation"])
        paragraph("Supporting references: "+", ".join(hypothesis["evidence_refs"]))
        for uncertainty in hypothesis["uncertainties"]:
            lines.append("- Unresolved: "+escape(uncertainty))
        lines.append("")
    else:
        paragraph("The available evidence does not establish a specific physical cause.")
    paragraph(report["caution"])
    heading("Next checks · recommended sources, not consulted")
    for check in report["ai"]["verification"]:
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
                        ("Detector checkpoint SHA-256",(result.get("analysis_settings") or {}).get("detector_sha256")),
                        ("Interpretation decoder",report["provenance"].get("decoder"))):
        lines.append(f"- {label}: {escape(value if value is not None else 'Not recorded')}")
    lines.append("")
    for warning in result.get("warnings") or []:
        paragraph("Acquisition warning: "+warning)
    paragraph(report["validation"])
    paragraph("No external web sources were retrieved for this report. Charts in the app use the counts and dB values listed above.")
    return "\n".join(lines).rstrip()+"\n"
