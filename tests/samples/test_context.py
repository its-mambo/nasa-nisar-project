"""Keep regional source provenance separate from immutable detector evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from nisar.app.paths import FRONTEND_ROOT, SAMPLE_ROOT
from nisar.app.web import STATIC_FILES
from nisar.features.samples.context import regional_context
from nisar.features.samples.store import CachedSample
from nisar.features.reporting.evidence_report import render_export


class RegionalContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = CachedSample(SAMPLE_ROOT)

    def test_offsets_units_and_source_snapshots_match_the_reviewed_case(self):
        result = self.sample.result()
        context, = result["evidence_report"]["external_context"]
        self.assertEqual([row["open_water_ha"] for row in context["observations"]], [6228, 3245, 2242])
        self.assertEqual([row["offset_days"] for row in context["observations"]], [6, -14, 10])
        self.assertEqual([row["nearest_radar_label"] for row in context["observations"]], ["Before", "After", "After"])
        self.assertIn("not completed", context["optical_verification_status"])
        for source in context["sources"]:
            path, _ = STATIC_FILES[source["image_url"].lstrip("/")]
            self.assertEqual(hashlib.sha256((FRONTEND_ROOT / path).read_bytes()).hexdigest(), source["image_sha256"])
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(source["url"].startswith("https://www.cawater-info.net/aral/data/pdf/"))

    def test_context_does_not_modify_saved_measurements_or_leak_into_other_cases(self):
        original = deepcopy(self.sample._result)
        first = self.sample.result()
        self.assertEqual(self.sample._result, original)
        for field in ("changed_pixels", "threshold", "roi", "paired_sector_deltas"):
            self.assertEqual(first[field], original[field])
        self.assertEqual(first["evidence_report"]["evidence"], original["evidence_report"]["evidence"])
        first["evidence_report"]["external_context"][0]["observations"].clear()
        self.assertEqual(len(self.sample.result()["evidence_report"]["external_context"][0]["observations"]), 3)
        live = self.sample.result(); live.pop("sample")
        self.assertEqual(regional_context(live), [])
        other = self.sample.result(); other["roi"]["t0_pixel_box"] = [0, 0, 128, 128]
        self.assertEqual(regional_context(other), [])
        other = self.sample.result(); other["acquisitions"]["t1"]["zeroDopplerStartTime"] = "2026-08-01"
        self.assertEqual(regional_context(other), [])

    def test_portable_exports_include_context_and_keep_attribution_unresolved(self):
        result = self.sample.result()
        self.assertIn("do not confirm", result["evidence_report"]["caution"])
        for format in ("md", "txt"):
            text = render_export(result, format)
            for phrase in ("62.28 km²", "32.45 km²", "22.42 km²", "6 days after Before",
                           "14 days before After", "10 days after After", "Table 3", "Source PDF SHA-256",
                           "Location-level comparison not completed", "do not validate individual flags"):
                self.assertIn(phrase.lower(), text.lower())
            self.assertNotIn("No external web sources were retrieved", text)
            self.assertNotIn("No external sources were consulted", text)


if __name__ == "__main__":
    unittest.main()
