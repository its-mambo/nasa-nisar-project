"""Build a bounded, ordered visual and measured context for Qwen."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mask_preview(flags, valid, edge=512):
    """Pool by presence so even a single flagged cell stays visible."""
    h, w = flags.shape
    ratio = min(1, edge / max(h, w))
    out_h, out_w = max(1, round(h * ratio)), max(1, round(w * ratio))
    rows, cols = np.arange(out_h) * h // out_h, np.arange(out_w) * w // out_w

    def pool(a):
        return np.maximum.reduceat(np.maximum.reduceat(a, rows, axis=0), cols, axis=1)

    return Image.fromarray(np.where(pool(flags & valid), 255, np.where(pool(valid), 0, 127)).astype("uint8"))


def snapshot(directory: Path, mask, result):
    """Read clean images under the analysis lock, without retaining full-size rasters."""
    images = []
    for name in ("t0.png", "t1.png"):
        with Image.open(directory / name) as source:
            source.thumbnail((512, 512), Image.Resampling.BILINEAR)
            images.append(source.convert("RGB"))
    mask_source = "detector_mask"
    if mask is None:
        # Older saved samples stored the display overlay, not the binary mask.
        # Nonzero alpha is only a subset of all originally flagged cells.
        with Image.open(directory / "overlay.png") as overlay:
            flags = np.asarray(overlay.getchannel("A")) > 0
        mask = mask_preview(flags, np.ones_like(flags))
        mask_source = "saved_overlay_footprint"
    images.append(mask.convert("RGB"))
    fields = ("zeroDopplerStartTime", "trackNumber", "frameNumber", "orbitPassDirection", "lookDirection")
    evidence = dict(
        mask_source=mask_source,
        acquisitions={key: {field: value.get(field) for field in fields}
                      for key, value in (result.get("acquisitions") or {}).items()},
        threshold=result.get("threshold"), changed_cells=result.get("changed_pixels"),
        paired_cells=result.get("valid_pixels"), spacing_m=(result.get("roi") or {}).get("model_spacing_m"),
        warnings=(result.get("warnings") or [])[:8],
        leading_sector_paired_t1_minus_t0_db=(result.get("paired_sector_deltas") or [])[:3],
    )
    return images, json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
