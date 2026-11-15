"""Verify that reports contain reproducible detector evidence and no old claims."""
import unittest

from nisar.app.paths import SAMPLE_ROOT
from nisar.features.reporting.evidence_report import build_report, render_export
from nisar.features.samples.store import CachedSample


class EvidenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = CachedSample(SAMPLE_ROOT)

    def test_saved_result_keeps_signed_measurements_and_drops_removed_outputs(self):
        raw = self.sample._result
        result = self.sample.result()
        for key in ("interpretation", "predictions", "candidate_regions", "pattern_evidence"):
            self.assertIn(key, raw)
            self.assertNotIn(key, result)
        self.assertEqual(result["paired_sector_deltas"], raw["paired_sector_deltas"])
        self.assertNotIn("Model label and interpretation", [s["title"] for s in result["report_sections"]])
        report = result["evidence_report"]
        by_id = {item["id"]: item for item in report["evidence"]}
        self.assertEqual(by_id["detector:summary"]["cells"], raw["changed_pixels"])
        for item in by_id.values():
            if item.get("sector"):
                entry = next(row for row in raw["paired_sector_deltas"] if row["zone"] == item["sector"])
                index = {"HH": 0, "HV": 1}[item["channel"]]
                self.assertEqual(item["value"], entry[item["group"]]["median_delta_hh_hv_db"][index])
        for ref in report["featured_evidence"] + report["finding"]["evidence_refs"]:
            self.assertIn(ref, by_id)

    def test_no_flags_and_acquisition_warnings_stay_visible(self):
        case = self.sample.result()
        case.update(changed_pixels=0, warnings=["Acquisition mismatch."])
        report = build_report(case)
        self.assertIn("No model cells were flagged", report["finding"]["text"])
        self.assertIn("comparability warnings", report["caution"])
        self.assertNotIn("ai", report)

    def test_exports_show_evidence_and_not_legacy_model_output(self):
        for format in ("md", "txt"):
            export = render_export(self.sample.result(), format)
            self.assertIn("Unflagged is not confirmed stable", export)
            self.assertIn("Detector checkpoint SHA-256", export)
            self.assertNotIn("Hypothesis", export)
            self.assertNotIn("Qwen", export)
