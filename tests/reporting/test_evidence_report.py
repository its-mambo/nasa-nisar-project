"""Report contracts and exports, without loading Qwen or detector weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from nisar.features.reporting.evidence_report import build_report, evidence_catalog, parse_interpretation, render_export
from nisar.app.paths import FRONTEND_ROOT
from nisar.features.analysis.service import NAMES
from nisar.app.server import create_app
from tests.support import fixture_demo, request


def structured_claims(result):
    reference=next(item["id"] for item in evidence_catalog(result) if item.get("sector") and item["group"]=="flagged")
    return {
        "observations":[{"text":"The radar return appears stronger in the After image.",
                         "evidence_refs":["image:t0","image:t1",reference]}],
        "hypothesis":{"title":"Possible surface-roughness change",
            "explanation":"A roughness difference could contribute to the changed returns.",
            "evidence_refs":["ai:observation:1"],
            "uncertainties":["Moisture and acquisition effects have not been independently checked."]},
        "verification":[{"source_type":"optical_imagery",
            "compare":"Compare the same flagged locations in optical images near both acquisition dates.",
            "supports":"A corresponding surface change would support further investigation.",
            "weakens":"Unchanged surfaces would weaken the roughness explanation."}],
    }


class EvidenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=tempfile.TemporaryDirectory()
        cls.demo=fixture_demo(cls.work.name)
        cls.base=cls.demo.infer([0,0,128,128],cls.demo.info["revision"],.1)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def result(self):
        result=copy.deepcopy(self.base)
        result["interpretation"]["summary"]=json.dumps(structured_claims(result))
        result.pop("evidence_report",None)
        return result

    def test_catalog_uses_actual_measurements_and_keeps_provenance(self):
        result=self.result()
        catalog=evidence_catalog(result)
        summary=next(item for item in catalog if item["id"]=="detector:summary")
        self.assertEqual(summary["cells"],result["changed_pixels"])
        self.assertEqual(summary["kind"],"detector")
        measurement=next(item for item in catalog if item.get("sector"))
        self.assertEqual(measurement["value"],result["paired_sector_deltas"][0]["flagged"]["median_delta_hh_hv_db"][0])
        self.assertEqual(measurement["kind"],"measured")
        self.assertTrue(measurement["source_paths"])

    def test_valid_claims_keep_ai_observations_and_hypotheses_separate(self):
        report=build_report(self.result())
        self.assertEqual(report["schema_version"],"nisar_evidence_report_v1")
        self.assertEqual(report["finding"]["kind"],"detector")
        self.assertEqual(report["ai"]["status"],"structured")
        self.assertEqual(report["ai"]["hypothesis"]["evidence_refs"],["ai:observation:1"])
        self.assertEqual(report["attribution_status"],"Unconfirmed hypothesis")
        self.assertFalse(report["ai"]["verification"][0]["consulted"])
        self.assertEqual(report["external_context"],[])

    def test_unknown_references_missing_uncertainty_and_extra_confidence_are_rejected(self):
        result=self.result()
        cases=[]
        payload=structured_claims(result); payload["observations"][0]["evidence_refs"]=["web:invented"]
        cases.append(payload)
        payload=structured_claims(result); payload["hypothesis"]["uncertainties"]=[]
        cases.append(payload)
        payload=structured_claims(result); payload["confidence"]=.99
        cases.append(payload)
        payload=structured_claims(result); payload["observations"].append(copy.deepcopy(payload["observations"][0]))
        cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload),self.assertRaises(ValueError):
                parse_interpretation(json.dumps(payload),evidence_catalog(result))

    def test_numeric_hallucinations_and_unretrieved_urls_are_rejected(self):
        result=self.result()
        for text in ("The return changed by -4.39 dB.","The cause is 90% certain.","An area of 25 km² changed.",
                     "There were 100 cells.","This was confirmed at https://example.com/news."):
            payload=structured_claims(result); payload["observations"][0]["text"]=text
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_interpretation(json.dumps(payload),evidence_catalog(result))

    def test_empty_or_legacy_model_output_preserves_a_useful_measured_report(self):
        for raw in ("","What the radar shows\nA very long legacy interpretation.","{broken json"):
            result=self.result(); result["interpretation"]["summary"]=raw
            report=build_report(result)
            self.assertEqual(report["ai"]["status"],"unavailable")
            self.assertIsNone(report["ai"]["hypothesis"])
            self.assertTrue(report["featured_evidence"])
            self.assertEqual(report["attribution_status"],"Cause unresolved")
            self.assertEqual(result["interpretation"]["summary"],raw)

    def test_no_flags_or_acquisition_warnings_leave_the_physical_cause_open(self):
        for warning,flags in (([],0),(["Track mismatch"],100)):
            result=self.result(); result["warnings"]=warning; result["changed_pixels"]=flags
            report=build_report(result)
            self.assertEqual(report["attribution_status"],"Cause unresolved")
            self.assertIsNone(report["ai"]["hypothesis"])
        result=self.result(); result["valid_pixels"]=result["changed_pixels"]=0
        self.assertIn("not enough jointly valid",build_report(result)["finding"]["text"])

    def test_sample_fallback_preserves_model_words_without_validated_claims(self):
        result = self.result()
        payload = structured_claims(result)
        del payload["hypothesis"]["uncertainties"]
        raw = json.dumps(payload)
        result["interpretation"]["summary"] = raw
        # The live workflow still rejects the same incomplete structured response.
        self.assertEqual(build_report(result)["ai"]["status"], "unavailable")
        report = build_report(result, allow_text_fallback=True)
        self.assertEqual(report["ai"]["status"], "text_fallback")
        self.assertIn(payload["observations"][0]["text"], report["ai"]["text"])
        self.assertIn(payload["hypothesis"]["explanation"], report["ai"]["text"])
        self.assertIn("not consulted", report["ai"]["text"])
        self.assertEqual(report["ai"]["observations"], [])
        self.assertIsNone(report["ai"]["hypothesis"])
        self.assertEqual(report["attribution_status"], "Cause unresolved")
        self.assertEqual(result["interpretation"]["summary"], raw)
        result["evidence_report"] = report
        for format in ("md", "txt"):
            export = render_export(result, format)
            self.assertIn(payload["observations"][0]["text"], export)
            self.assertIn("unverified original model text", export.lower())
            self.assertNotIn("A usable structured AI interpretation is unavailable", export)

    def test_sample_fallback_keeps_numeric_url_and_non_model_rejections(self):
        result = self.result()
        for text in ("A change of 90% is certain.", "Confirmed at https://example.com", "{broken json", ""):
            result["interpretation"]["summary"] = text
            self.assertEqual(build_report(result, allow_text_fallback=True)["ai"]["status"], "unavailable")
        result["interpretation"] = {"source": "deterministic", "summary": "A template explanation."}
        self.assertEqual(build_report(result, allow_text_fallback=True)["ai"]["status"], "unavailable")

    def test_sample_fallback_does_not_show_hypotheses_with_flags_or_geometry_issues(self):
        for warnings, flags in ((["Track mismatch"], 100), ([], 0)):
            result = self.result()
            payload = structured_claims(result)
            del payload["hypothesis"]["uncertainties"]
            result["interpretation"]["summary"] = json.dumps(payload)
            result["warnings"], result["changed_pixels"] = warnings, flags
            report = build_report(result, allow_text_fallback=True)
            self.assertEqual(report["ai"]["status"], "text_fallback")
            self.assertNotIn(payload["hypothesis"]["explanation"], report["ai"]["text"])
            self.assertIsNone(report["ai"]["hypothesis"])

    def test_plain_prose_fallback_and_markdown_escaping(self):
        result = self.result()
        raw = "The basin looks darker.\n\n<script>alert('claim')</script> # MODEL TEXT"
        result["interpretation"]["summary"] = raw
        report = build_report(result, allow_text_fallback=True)
        self.assertEqual(report["ai"]["status"], "text_fallback")
        result["evidence_report"] = report
        self.assertIn(r"\<script\>", render_export(result, "md"))

    def test_markdown_and_text_preserve_values_labels_and_verification_scope(self):
        result=self.result(); result["evidence_report"]=build_report(result)
        markdown=render_export(result,"md"); plain=render_export(result,"txt")
        self.assertTrue(markdown.startswith("# Evidence & interpretation"))
        self.assertTrue(plain.startswith("EVIDENCE & INTERPRETATION"))
        for text in (markdown,plain):
            self.assertIn("unconfirmed hypothesis",text.lower())
            self.assertIn("not consulted",text.lower())
            self.assertIn("No external web sources were retrieved",text)
            self.assertIn("16,384",text)
            self.assertIn("EPSG:32640",text)
            self.assertIn("2026-06-21",text)
            self.assertIn("After-minus-Before",text)
            self.assertNotIn("/t0.png?v=",text)
        with self.assertRaises(ValueError):
            render_export(result,"pdf")

    def test_markdown_treats_model_formatting_as_text(self):
        result=self.result(); payload=structured_claims(result)
        payload["observations"][0]["text"]="<script>alert('claim')</script> # MEASURED"
        result["interpretation"]["summary"]=json.dumps(payload)
        markdown=render_export(result,"md")
        self.assertIn(r"\<script\>",markdown)
        self.assertIn(r"\# MEASURED",markdown)


class ReportExportApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_revision_checked_markdown_text_and_legacy_upgrade(self):
        with tempfile.TemporaryDirectory() as work:
            demo=fixture_demo(work)
            result=demo.infer([0,0,128,128],demo.info["revision"],.1)
            result["interpretation"]["summary"]=json.dumps(structured_claims(result))
            result.pop("evidence_report")
            app=create_app(demo,FRONTEND_ROOT,NAMES)
            for extension in ("md","txt"):
                status,body=await request(app,"GET",f"/report.{extension}?v={demo.result_revision}")
                self.assertEqual(status,200)
                self.assertIn(b"unconfirmed hypothesis",body.lower())
                self.assertIn(b"Detector output",body)
            status,body=await request(app,"GET","/result")
            self.assertEqual(json.loads(body)["result"]["evidence_report"]["ai"]["status"],"structured")
            status,_=await request(app,"GET","/report.md?v=old")
            self.assertEqual(status,410)
            status,_=await request(app,"GET","/report.pdf")
            self.assertEqual(status,404)


if __name__=="__main__":
    unittest.main()
