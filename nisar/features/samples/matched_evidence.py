"""Attach the separately reviewed optical pilot to its exact saved radar result."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json

from nisar.app.paths import PROJECT_ROOT

SITE_IDS = ("contraction", "reference", "bright-surface", "quality-blocked")
IMAGE_FILES = frozenset(
    f"{site}-{kind}.png" for site in SITE_IDS
    for kind in ("radar_before", "radar_after", "overlay", "before-natural", "after-natural",
                 "before-false_color", "after-false_color")
)


@lru_cache(maxsize=1)
def pilot_record():
    path = PROJECT_ROOT / "evaluation/karakalpakstan-pilot/manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for relative, checksum in record["artifact_sha256"].items():
        source = (PROJECT_ROOT / relative).resolve()
        if not source.is_relative_to(PROJECT_ROOT):
            raise ValueError("Invalid matched-evidence artifact path")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != checksum:
            raise ValueError("Matched-evidence checksum mismatch: " + relative)
    return record


def matched_evidence(result):
    roi = result.get("roi") or {}
    sample = result.get("sample") or {}
    acquisitions = result.get("acquisitions") or {}
    if (sample.get("revision") != "sample-0893b8c3af904967"
            or roi.get("epsg") != 32640
            or roi.get("t0_pixel_box") != [5323, 2414, 11255, 6066]
            or result.get("threshold") != .1
            or [(acquisitions.get(k) or {}).get("zeroDopplerStartTime", "")[:10] for k in ("t0", "t1")]
               != ["2026-06-21", "2026-09-13"]):
        return None
    return deepcopy(pilot_record())
