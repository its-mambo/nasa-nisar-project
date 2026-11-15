"""Check location, source, measurement and scope integrity of the optical pilot."""
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from nisar.app.paths import PROJECT_ROOT, SAMPLE_ROOT
from nisar.app.web import STATIC_FILES
from nisar.features.samples.matched_evidence import matched_evidence
from nisar.features.samples.store import CachedSample
from nisar.features.reporting.evidence_report import render_export


class MatchedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = CachedSample(SAMPLE_ROOT)
        cls.result = cls.sample.result()
        cls.pilot = cls.result["evidence_report"]["matched_evidence"]

    def test_sources_are_near_radar_dates_and_all_artifacts_are_portable(self):
        self.assertEqual([s["offset_days"] for s in self.pilot["sources"]], [-3, 1])
        for source in self.pilot["sources"]:
            self.assertEqual((date.fromisoformat(source["date"]) - date.fromisoformat(source["radar_date"])).days, source["offset_days"])
        for relative, expected in self.pilot["artifact_sha256"].items():
            with self.subTest(artifact=relative), (PROJECT_ROOT / relative).open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), expected)
        for site in self.pilot["sites"]:
            for url in (*site["images"].values(), *site["source_crops"].values()):
                self.assertIn(url.lstrip("/"), STATIC_FILES)

    def test_real_crop_measurements_match_raw_arrays_and_saved_overlay(self):
        west, south, east, north = self.result["roi"]["projected_bounds_wsen"]
        with Image.open(self.sample.directory / "overlay.png") as image:
            for site in self.pilot["sites"]:
                with self.subTest(site=site["id"]):
                    w, s, e, n = site["bounds_wsen"]
                    self.assertTrue(west <= w < e <= east and south <= s < n <= north)
                    self.assertEqual([e - w, n - s], [3000, 3000])
                    box = site["sample_pixel_box"]
                    self.assertEqual([west + box[0] * 20, north - box[3] * 20,
                                      west + box[2] * 20, north - box[1] * 20], [w, s, e, n])
                    arrays = np.load(PROJECT_ROOT / f"evaluation/karakalpakstan-pilot/source/{site['id']}-radar.npz")
                    valid = arrays["paired_valid"]
                    expected = np.median((arrays["t1_db"] - arrays["t0_db"])[:, valid], axis=1)
                    np.testing.assert_allclose(expected, site["radar"]["median_delta_hh_hv_db"], atol=1e-6)
                    np.testing.assert_array_equal(arrays["overlay_alpha"], np.asarray(image.getchannel("A").crop(box)))
                    self.assertEqual(int((arrays["overlay_alpha"][valid] > 0).sum()), site["radar"]["visible_flag_cells"])
                    self.assertEqual(int(valid.sum()), site["radar"]["paired_cells"])

    def test_pilot_cannot_be_reused_as_other_result_or_mutate_detector_measurements(self):
        original = deepcopy(self.sample._result)
        for field in ("changed_pixels", "valid_pixels", "threshold", "roi", "paired_sector_deltas"):
            self.assertEqual(self.result[field], original[field])
        mutations = [
            ("sample", {}), ("threshold", .2), ("roi", {"epsg": 32640, "t0_pixel_box": [0, 0, 10, 10]}),
            ("acquisitions", {}),
        ]
        for key, value in mutations:
            changed = {**self.result, key: value}
            self.assertIsNone(matched_evidence(changed))
        first = matched_evidence(self.result)
        first["sites"].clear()
        self.assertEqual(len(matched_evidence(self.result)["sites"]), 4)

    def test_export_preserves_interpretation_failure_quality_failure_and_no_accuracy_claim(self):
        self.assertEqual([s["outcome"] for s in self.pilot["sites"]],
                         ["optical_support", "comparison", "interpretation_failure", "unresolved"])
        self.assertLess(self.pilot["sites"][3]["optical"]["after"]["usable_fraction"], .1)
        self.assertEqual(self.pilot["sites"][2]["optical"]["before"]["source_water_flag_fraction"], 0)
        point = self.pilot["sites"][0]["comparison_point"]
        self.assertEqual([point["easting_m"], point["northing_m"]], [626940, 4808910])
        self.assertEqual(point["optical"]["before"]["source_water_flag_cells"], 1)
        self.assertEqual(point["optical"]["after"]["source_water_flag_cells"], 0)
        self.assertLess(point["optical"]["after"]["median_mndwi"], 0)
        for format in ("md", "txt"):
            text = render_export(self.result, format)
            for phrase in ("water-loss interpretation fails", "not a proven false positive", "three days before",
                           "not a detector accuracy benchmark", "lower bound", "no independent human or field review"):
                self.assertIn(phrase.lower(), text.lower())
            self.assertIn("626000", text)
            self.assertIn("626940 E", text)
            self.assertIn("2026-09-14", text)


if __name__ == "__main__":
    unittest.main()
