"""Build the same image and measured-evidence prompt for base and adapted Qwen."""
import json

from roi_report import acquisition_date
from roi_visual import PANEL_NAMES


SYSTEM = """Interpret paired NISAR HH/HV radar images for a researcher.
Panel 1 is clean T0 and panel 2 is clean T1. Compare the same place in both.
Panel 3 is a standalone detector map, with no radar intensity: white marks detector
flags, black marks valid unflagged areas, and gray marks no paired data or padding.
At reduced resolution, one white pixel can represent several cells with at least
one flag. Mask colors do not encode increase, decrease, land cover, or cause.
Ignore black padding in the clean radar panels.

Use the mask for location and shape. Describe what is visible in clean T0 and T1.
Use the signed, paired T1-minus-T0 HH/HV medians for direction in the matching
sector: negative means weaker, positive means stronger. These are sector group
statistics, not measurements of an individual patch. Compare flagged and
unflagged cells within the same sector. If the channels disagree, report mixed
direction; if the evidence does not resolve direction at the patch, say so.

Offer a specific physical hypothesis only when you can name a concrete feature
in the clean pair or a measured contrast that supports it. Describe scattered
patches as scattered; do not infer a continuous change from sector totals.
A similar outline with weaker interior returns does not show water expansion.
If no specific mechanism is supported, say what the evidence cannot distinguish.
Do not use vague substitutes like "different surface conditions." With no
flags or an acquisition comparability warning, leave physical attribution open.

For "How to check", name an independent source, what to compare at the flagged
location near both dates, and what result would weaken the hypothesis. If the
cause is unresolved, name a check that distinguishes plausible alternatives.
Do not just repeat the flagged-versus-unflagged comparison already supplied.

Use three short plain-text sections: What the radar shows, Possible explanation,
and How to check. The measured report already lists the numbers."""


def messages(result):
    acquisitions=result.get("acquisitions") or {}
    content=[]
    for i,name in enumerate(PANEL_NAMES):
        if i<2:
            meta=acquisitions.get(f"t{i}") or {}
            date=acquisition_date(meta.get("zeroDopplerStartTime"))
            if date:
                name+=f" — {date}"
        content.extend(({"type":"text","text":f"\nPanel {i+1}: {name}\n"},{"type":"image"}))

    evidence={
        "total_flagged_cells":result["changed_pixels"],
        "total_valid_cells":result["valid_pixels"],
        "paired_sector_deltas_db_T1_minus_T0_HH_then_HV":result.get("paired_sector_deltas",[]),
        "acquisition_warnings":result.get("warnings") or [],
    }
    content.append({"type":"text","text":(
        "\nMeasured evidence (null deltas mean insufficient samples):\n"
        +json.dumps(evidence,ensure_ascii=False,allow_nan=False)
        +"\nDescribe the main supported change and what might explain it."
    )})
    return [{"role":"system","content":SYSTEM},{"role":"user","content":content}]
