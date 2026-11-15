"""Reviewed context attached at playback, separately from immutable inference."""
from datetime import date
import json
from pathlib import Path


def regional_context(result):
    roi = result.get("roi") or {}
    acquisitions = result.get("acquisitions") or {}
    stamps = [(acquisitions.get(key) or {}).get("zeroDopplerStartTime", "")[:10]
              for key in ("t0", "t1")]
    if (not result.get("sample") or roi.get("epsg") != 32640
            or roi.get("t0_pixel_box") != [5323, 2414, 11255, 6066]
            or stamps != ["2026-06-21", "2026-09-13"]):
        return []
    record = json.loads(Path(__file__).with_name("karakalpakstan_context.json").read_text(encoding="utf-8"))
    radar_dates = [date.fromisoformat(stamp) for stamp in stamps]
    for row in [*record["observations"], *record["sources"]]:
        observed = date.fromisoformat(row.get("date") or row["observation_date"])
        index = min(range(2), key=lambda i: abs((observed - radar_dates[i]).days))
        row["nearest_radar_date"] = stamps[index]
        row["nearest_radar_label"] = ("Before", "After")[index]
        row["offset_days"] = (observed - radar_dates[index]).days
    return [record]
