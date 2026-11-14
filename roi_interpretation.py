"""Extract change-pattern evidence from aligned HH/HV rasters and the detector mask."""
import numpy as np


LIMIT = ("These are hypotheses from two SAR acquisitions. Weak returns are not a water map, "
         "and the detector and linear label do not identify a physical cause.")
ZONES = ("upper left","upper center","upper right","middle left","center",
         "middle right","lower left","lower center","lower right")
UNSURE = {
    "id": "unclear",
    "title": "Cause unclear",
    "possibility": "Report this as a radar backscatter change. The pattern does not justify naming one cause.",
    "check": "Compare the flagged areas with a cloud-free optical image, a water or land-cover map, or another radar date.",
}


def connected_support(grid, seeds=None):
    """Retain groups of >=3 blocks, optionally touching a contextual seed."""
    h, w = grid.shape
    seen = np.zeros_like(grid)
    keep, largest = np.zeros_like(grid), 0
    for y, x in zip(*np.nonzero(grid)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo, component = [(y, x)], []
        while todo:
            r, c = todo.pop()
            component.append((r,c))
            for a, b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= a < h and 0 <= b < w and grid[a,b] and not seen[a,b]:
                    seen[a,b] = True
                    todo.append((a,b))
        if len(component) >= 3 and (seeds is None or any(seeds[r,c] for r,c in component)):
            rows, cols = zip(*component)
            keep[rows,cols] = True
            largest = max(largest,len(component))
    return keep, largest


def near(grid):
    padded = np.pad(grid, 1)
    h, w = grid.shape
    return np.logical_or.reduce([padded[y:y+h,x:x+w] for y in range(3) for x in range(3)])


def radar_patterns(t0, t1, valid, mask):
    """Count patterns on the model grid. Temp arrays are limited to one strip."""
    h, w = valid.shape
    block = max(4, (max(h, w) + 127)//128)
    xs = np.arange(0,w,block)
    ys = np.arange(0,h,block)
    cell_area = np.minimum(block,h-ys)[:,None]*np.minimum(block,w-xs)
    step = max(1, (max(h,w) + 319)//320)
    sample = valid[::step,::step]
    # Small valid footprints can fall between systematic sample points.
    if int(sample.sum()) < 32:
        return {"status":"insufficient_samples","candidates":[]}
    samples = np.concatenate([t[:,::step,::step][:,sample] for t in (t0,t1)],axis=1)
    low, middle = np.quantile(samples,[.25,.5],axis=1)
    hh,hv = np.minimum(low,middle-[1.5,1.0])
    keys = ("valid","flags","weak_t0","weak_t1","new_weak","lost_weak","stable_weak",
            "brighter_up","brighter_down")
    counts = {k:np.zeros(cell_area.shape,np.int64) for k in keys}
    bg_up = bg_down = 0
    for i, y in enumerate(ys):
        s = slice(y,y+block)
        v = valid[s]
        f = mask[s] & v
        weak0 = (t0[0,s] <= hh) & (t0[1,s] <= hv)
        weak1 = (t1[0,s] <= hh) & (t1[1,s] <= hv)
        dhh, dhv = t1[0,s]-t0[0,s], t1[1,s]-t0[1,s]
        darkens = (dhh <= -3) & (dhv <= -2)
        brightens = (dhh >= 3) & (dhv >= 2)
        bg_up += int(np.count_nonzero(v & ~f & brightens))
        bg_down += int(np.count_nonzero(v & ~f & darkens))
        brighter = ~weak0 & ~weak1
        for k, pixels in zip(keys, (v,f,v&weak0,v&weak1,f&~weak0&weak1&darkens,
                                    f&weak0&~weak1&brightens,f&weak0&weak1,
                                    f&brighter&brightens,f&brighter&darkens), strict=True):
            counts[k][i] = np.add.reduceat(pixels.sum(axis=0),xs)

    n, f = counts["valid"], counts["flags"]
    total = int(f.sum())
    unflagged = int(n.sum()) - total
    widespread = set()
    if unflagged >= 32:
        if bg_up / unflagged >= .35:
            widespread.update(("lost_weak","brighter_up"))
        if bg_down / unflagged >= .35:
            widespread.update(("new_weak","brighter_down"))
    good = (n >= .75*cell_area) & (n >= 8)
    old_core = good & (counts["weak_t0"] >= .35*n)
    new_core = good & (counts["weak_t1"] >= .35*n)
    patterns = {
        "darkening_at_weak_area": (("new_weak",), near(old_core)),
        "brightening_at_weak_area": (("lost_weak",), near(new_core)),
        "persistently_weak": (("stable_weak",), None),
        "brighter_area": (("brighter_up","brighter_down"), None),
    }
    zone_idx = 3*(3*np.minimum(ys+block//2,h-1)//h)[:,None] + (3*np.minimum(xs+block//2,w-1)//w)
    measurements = {}
    for k,(components,context) in patterns.items():
        supported, largest = np.zeros_like(n), 0
        for component in components:
            if component in widespread:
                continue  # The same dB shift is common even outside the flags.
            direction = counts[component]
            patch = good & (direction >= 4) & (direction >= .15*n) & (direction >= .5*f)
            keep, size = connected_support(patch,context)
            supported[keep] += direction[keep]
            largest = max(largest,size)
        cells = int(supported.sum())
        zones = np.bincount(zone_idx.ravel(),weights=supported.ravel(),minlength=9)
        measurements[k] = {"flagged_cells":cells, "largest_cluster_blocks":largest,
                           "leading_zone":ZONES[zones.argmax()] if cells else None}

    candidates = []
    def add(key, title, evidence, possibility, check, fraction):
        measure = measurements[key]
        cells = measure["flagged_cells"]
        cluster = measure["largest_cluster_blocks"]
        if cells >= max(32,fraction*total) and cluster >= 3:
            candidates.append({
                "id":key, "title":title,
                "evidence":f"{evidence} This pattern accounts for {cells:,} of {total:,} flagged cells "
                           f"({cells/total:.1%}), with the largest share in the {measure['leading_zone']} of the selection.",
                "possibility":possibility, "check":check,
            })

    add("darkening_at_weak_area","Possible surface-water expansion",
        "Flagged patches darkened in both polarizations and became weak near areas that were already weak at T0.",
        "If the earlier dark areas were water, this could be a rising shoreline or inundation. "
        "Wetter or smoother ground can also change the return.",
        "Compare the patch edges against a water map and optical images from both dates.", .12)
    add("brightening_at_weak_area","Possible surface-water retreat or exposure",
        "Flagged patches brightened in both polarizations near areas that remained weak at T1.",
        "If these patches were previously water-covered, the pattern could reflect water retreat or exposed ground. "
        "A change in surface roughness is another possibility.",
        "Look for a consistent shoreline shift across optical imagery or additional radar dates.", .12)
    add("persistently_weak","Possible change in radar returns over an already dark surface",
        "Flagged patches remained weak in both HH and HV at both dates.",
        "If the patches are open water, wind-driven roughness or noise in weak radar returns are plausible. "
        "The pattern does not show water expansion by itself.",
        "Check whether the flags sit inside an unchanged water boundary and compare with another acquisition.", .25)
    add("brighter_area","Possible land-surface change",
        "Outside the persistently low-return pattern, flagged patches contain substantial changes in both polarizations, "
        "including a connected area with changes in the same direction.",
        "If the patches coincide with fields, crop growth, harvest, or soil moisture changes could explain them. "
        "The radar pattern alone does not identify crops.",
        "Check whether the flags follow field boundaries, then compare vegetation and ground conditions on both dates.", .22)

    return {"status":"measured","flagged_cells_analyzed":total,"block_size_cells":block,
            "weak_cutoffs_hh_hv_db":[round(float(hh),2),round(float(hv),2)],
            "widespread_unflagged_shift":bool(widespread),
            "patterns":measurements,"candidates":candidates}


def interpret(evidence, result, ranker):
    """Show only evidence-backed options. Qwen ranks; fixed wording keeps claims grounded."""
    candidates = evidence["candidates"]
    unclear = {**UNSURE, "evidence": (
        "No tested pattern had enough spatial support to narrow the cause." if not candidates
        else "The measured patterns can have multiple physical explanations."
    )}
    options = [*candidates, unclear]
    source = "rules"
    notes = result.get("warnings", [])
    prediction = result.get("predictions", {}).get("linear")
    # Don't name direction-dependent causes if the time order is reversed.
    # Likewise, avoid a primary cause when acquisition geometry is mismatched.
    if evidence["status"] == "insufficient_samples":
        unclear["evidence"] = "Too few jointly valid samples to characterize the radar pattern."
    elif not evidence["flagged_cells_analyzed"]:
        unclear["evidence"] = "No jointly valid detector flags were available to interpret."
    elif notes:
        unclear["evidence"] = "The acquisition checks raised a comparability or time-order warning."
        candidates = []
    elif evidence.get("widespread_unflagged_shift"):
        unclear["evidence"] = ("Many unflagged cells also shifted in the same direction in both channels. "
                               "A broad shift makes a local physical explanation harder to isolate.")
    order = ["unclear"]
    if candidates:
        facts = {"linear_label": prediction["label"] if prediction else None,
                 "label_note":"Experimental broad label only; it does not establish cause.",
                 "flagged_cells_analyzed":evidence["flagged_cells_analyzed"],
                 "options":[{k:v for k,v in o.items() if k != "id"} for o in options]}
        try:
            ranked = ranker(facts)
        except (RuntimeError, ValueError, KeyError, ImportError, TypeError, AttributeError):
            ranked = []
        if sorted(ranked) == list(range(len(options))):
            order = [options[i]["id"] for i in ranked]
            source = "qwen"
        else:
            order += [o["id"] for o in candidates]
    by_id = {o["id"]:o for o in options}
    return {"source":source, "primary":by_id[order[0]],
            "alternatives":[by_id[k] for k in order[1:] if k != "unclear"][:2],
            "limitation":LIMIT}
