"""Build readable, deterministic report sections from ROI measurements."""
from datetime import datetime, timezone
import numpy as np


def paired_sector_deltas(z0,z1,valid,mask,zones):
    """Summarize paired T1 minus T0 values on the detector grid, not date medians."""
    h,w=valid.shape
    leading=sorted((i for i,z in enumerate(zones) if z["flagged"]),
                   key=lambda i:-zones[i]["flagged"])[:3]
    comparisons=[]
    for i in leading:
        r,c=divmod(i,3)
        s=np.s_[r*h//3:(r+1)*h//3,c*w//3:(c+1)*w//3]
        good,flags=valid[s],mask[s]
        pair=z1[:,s[0],s[1]]-z0[:,s[0],s[1]]
        groups={}
        for name,selector in (("flagged",good&flags),("unflagged",good&~flags)):
            n=int(selector.sum())
            groups[name]={"cells":n,"median_delta_hh_hv_db":
                          [round(float(x),2) for x in np.median(pair[:,selector],axis=1)] if n>=32 else None}
        comparisons.append({"zone":zones[i]["zone"],**groups})
    return comparisons


def acquisition_date(value):
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z","+00:00"))
        return stamp.astimezone(timezone.utc).date() if stamp.tzinfo else stamp.date()
    except ValueError:
        return None


def report_sections(r):
    roi, acq = r["roi"], r["acquisitions"]
    sections = []
    def add(title,text):
        sections.append({"title":title,"text":text})

    dates = [acquisition_date(acq[k].get("zeroDopplerStartTime")) for k in ("t0","t1")]
    coverage = (f"The selection covers {roi['width_px']:,} × {roi['height_px']:,} native pixels, "
                f"or {roi['projected_rectangle_km2']:,.2f} km² on the projected grid (EPSG:{roi['epsg']}). "
                f"Both dates have usable HH and HV at {roi['native_joint_valid_pixels']:,} native pixels. "
                f"The detector works at {roi['model_spacing_m']:g} m spacing.")
    dates_text = ""
    if all(dates):
        days = (dates[1]-dates[0]).days
        dates_text = f"T0 was acquired on {dates[0]} and T1 on {dates[1]}."
        if days >= 0:
            dates_text += f" The acquisitions are {days} calendar {'day' if days == 1 else 'days'} apart."
    else:
        dates_text = " ".join(f"{k.upper()} was acquired on {d}." for k,d in zip(("t0","t1"),dates) if d)
    geometry = []
    for date in ("t0","t1"):
        m = acq[date]
        values = [str(m[k]) for k in ("orbitPassDirection","lookDirection") if k in m]
        values += [f"track {m['trackNumber']}"] if "trackNumber" in m else []
        values += [f"frame {m['frameNumber']}"] if "frameNumber" in m else []
        if values:
            geometry.append(f"{date.upper()}: {', '.join(values)}.")
    add("Acquisitions and coverage"," ".join(filter(None,[dates_text,*geometry,coverage])))

    def channels(values):
        return f"HH {values[0]:g} dB and HV {values[1]:g} dB"
    m0,m1 = r["roi_measurements"]["median_hh_hv_db_t0_t1"]
    add("Radar returns within the selection",
        f"Median backscatter at T0 was {channels(m0)}; at T1 it was {channels(m1)}. "
        f"The T1 minus T0 difference between date medians was HH {m1[0]-m0[0]:+.2f} dB "
        f"and HV {m1[1]-m0[1]:+.2f} dB. "
        + r["spatial_context"]["scene_description"])
    n,c = r["valid_pixels"],r["changed_pixels"]
    area = c * roi["model_spacing_m"]**2 / 1e6
    text = (f"At threshold {r['threshold']:g}, the detector flagged {c:,} of {n:,} valid model pixels "
            f"({c/n:.1%}), equivalent to approximately {area:,.2f} km² of flagged grid cells.") if n else "No jointly valid model pixels."
    active = sorted((z for z in r["spatial_context"]["zones"] if z["flagged"]),key=lambda z:-z["flagged"])
    if c and active:
        descriptions = [
            f"{z['zone']}: {z['flagged']:,} flags ({z['flagged']/c:.1%} of all flags; "
            f"{z['flagged']/z['valid']:.1%} of valid pixels within that sector)"
            for z in active[:3]
        ]
        text += " The sectors with the most flags were " + "; ".join(descriptions) + "."
    add("Where the detector flagged change",text)
    pairs=[]
    for row in r.get("paired_sector_deltas",[]):
        flagged=row["flagged"]["median_delta_hh_hv_db"]
        background=row["unflagged"]["median_delta_hh_hv_db"]
        if flagged is not None and background is not None:
            pairs.append(f"{row['zone']}: flagged HH {flagged[0]:+.2f} dB, HV {flagged[1]:+.2f} dB; "
                         f"unflagged HH {background[0]:+.2f} dB, HV {background[1]:+.2f} dB.")
    if pairs:
        add("Flagged patches versus background",
            "Median of paired T1 minus T0 backscatter on the detector grid. "+" ".join(pairs))

    full = r["full_scene_context"]
    if full["sample_pairs"]:
        add("Context from the common footprint",
            f"Across {full['sample_pairs']:,} matched samples from the shared HDF5 footprint, "
            f"the T0 medians were {channels(full['median_t0_hh_hv_db'])}; the T1 medians were "
            f"{channels(full['median_t1_hh_hv_db'])}. These are background statistics; "
            "the detector ran only inside the selected area.")
    text = "Radar flags indicate changed backscatter and do not identify a physical cause."
    if r["warnings"]:
        text += " " + " ".join(r["warnings"])
    add("Interpretation limits", text)
    return sections

def scene_observations(zones):
    """Summarize the scene and map in image coordinates, independent of overlay opacity."""
    valid_total = sum(z["valid"] for z in zones)
    flagged_total = sum(z["flagged"] for z in zones)
    context = {"grid_reference": "image frame", "zones": zones}
    if not valid_total:
        return {**context, "scene_description": "The input pair has no pixels valid at both dates.",
                "change_map_description": "The change map has no valid pixels to describe."}
    populated = [z for z in zones if z["valid"]]
    low = min(populated, key=lambda z: z["mean_hh_db"])
    high = max(populated, key=lambda z: z["mean_hh_db"])
    if len(populated) == 1:
        scene = f"Coverage shared by both dates lies within the {high['zone']} of the image frame."
    elif high["mean_hh_db"] - low["mean_hh_db"] >= 1:
        scene = (f"Across both dates, average HH radar returns were strongest in the "
                 f"{high['zone']} and weakest in the {low['zone']} of the image frame.")
    else:
        scene = "The two SAR images have overlapping coverage, with no strong contrast between the measured sector averages."
    if flagged_total:
        active = sorted((z for z in zones if z["flagged"]), key=lambda z: -z["flagged"])
        top = active[0]
        tied = sum(z["flagged"] == top["flagged"] for z in active)
        change = f"The detector flagged {flagged_total / valid_total:.1%} of valid pixels."
        if tied == 1:
            share = top["flagged"] / flagged_total
            change += (f" The {top['zone']} of the image frame had the most flagged pixels "
                       f"({share:.1%} of all flagged pixels).")
        else:
            change += " Multiple image sectors tied for the most flagged pixels."
    else:
        change = "The change map has no flagged pixels in the overlapping coverage."
    return {**context, "scene_description": scene, "change_map_description": change}


