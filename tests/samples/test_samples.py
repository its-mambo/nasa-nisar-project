import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nisar.app.launcher import choose_mode
from nisar.app.paths import FRONTEND_ROOT, PROJECT_ROOT
from nisar.app.server import create_app
from nisar.features.analysis.artifacts import NAMES
from nisar.features.samples.store import CachedSample, FORMAT, IMAGE_FILES, SampleSession, digest, load_sample
from nisar.features.samples.builder import publish
from tests.support import fixture_demo, request


def sample_fixture(root):
    root = Path(root)
    with tempfile.TemporaryDirectory() as work:
        demo = fixture_demo(work)
        result = demo.infer([0, 0, 128, 128], demo.info["revision"], .1)
        revision = "sample-0123456789abcdef"
        bundle = root / revision
        bundle.mkdir()
        for name in IMAGE_FILES:
            (bundle / name).write_bytes((Path(work) / ("t0.png" if name == "sample-preview.png" else name)).read_bytes())
        (bundle / "result.json").write_text(json.dumps(result))
    metadata = dict(roi=[0, 0, 128, 128], threshold=.1, sources={
        date: {"name": f"{date}.h5", "path": f"src/{date}.h5", "sha256": "abc"} for date in ("t0", "t1")},
        generated_at="2026-11-14T23:00:00+00:00", native_size_px=[128,128],
        projected_area_km2=1.6384, epsg=32640)
    files = {name: digest(bundle / name) for name in (*IMAGE_FILES, "result.json")}
    (bundle / "manifest.json").write_text(json.dumps(dict(format=FORMAT, revision=revision, sample=metadata, files=files)))
    (root / "current.json").write_text(json.dumps({"revision": revision}))
    return CachedSample(root)


class SampleStoreTests(unittest.TestCase):
    def test_publish_validates_inputs_and_preserves_previous_cache_on_failure(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as work:
            demo = fixture_demo(work)
            result = demo.infer([0, 0, 128, 128], demo.info["revision"], .1)
            report = Path(work) / "report.json"
            report.write_text(json.dumps(result))
            with patch("nisar.features.samples.builder.ROI", [0, 0, 128, 128]), patch(
                    "nisar.features.samples.builder.SOURCE_FILES", {"t0": "t0.h5", "t1": "t1.h5"}):
                sample = publish(work, work, root)
                self.assertTrue((sample.directory / "sample-preview.png").is_file())
                self.assertEqual(sample.metadata["sources"]["t0"]["sha256"], digest(Path(work) / "t0.h5"))
                pointer = (Path(root) / "current.json").read_bytes()
                result["threshold"] = .2
                report.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "threshold 0.10"):
                    publish(work, work, root)
                self.assertEqual((Path(root) / "current.json").read_bytes(), pointer)
                self.assertEqual(CachedSample(root).revision, sample.revision)

    def test_integrity_missing_cache_and_immutable_result(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(load_sample(root))
            sample = sample_fixture(root)
            first = sample.result()
            first["sample"]["roi"][0] = 99
            first["images"].clear()
            self.assertEqual(sample.result()["sample"]["roi"][0], 0)
            self.assertIn(sample.revision, sample.result()["images"]["t0.png"])
            (sample.directory / "t0.png").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                CachedSample(root)

    def test_invalid_revision_cannot_escape_cache_directory(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "current.json").write_text(json.dumps({"revision": "../../etc"}))
            with self.assertRaisesRegex(ValueError, "Invalid cached sample revision"):
                CachedSample(root)

    def test_choice_defaults_to_sample_and_accepts_yes(self):
        self.assertEqual(choose_mode(lambda _: ""), "sample")
        self.assertEqual(choose_mode(lambda _: "n"), "sample")
        self.assertEqual(choose_mode(lambda _: "Yes"), "model")

    def test_sample_startup_does_not_import_model_or_data_libraries(self):
        script = """
import builtins
original = builtins.__import__
blocked = {'torch', 'numpy', 'h5py', 'transformers', 'peft', 'safetensors'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Sample startup imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import nisar.app.launcher
import nisar.app.server
from nisar.features.samples.store import SampleSession
"""
        subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True, timeout=10)


class SampleApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_playback_exports_and_read_only_guards(self):
        with tempfile.TemporaryDirectory() as root:
            sample = sample_fixture(root)
            app = create_app(SampleSession(sample), FRONTEND_ROOT, NAMES, sample=sample, model_enabled=False)
            status, body = await request(app, "GET", "/sample")
            self.assertEqual(status, 200)
            capabilities = json.loads(body)
            self.assertFalse(capabilities["model_enabled"])
            self.assertTrue(capabilities["sample"]["available"])
            for path in ("/result", f"/result?v={sample.revision}"):
                status, body = await request(app, "GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["result"]["sample"]["revision"], sample.revision)
            for path in ("/infer", "/prepare", "/upload/t0"):
                status, body = await request(app, "POST", path, {})
                self.assertEqual(status, 409)
                self.assertIn(b"read-only", body)
            for extension in ("md", "txt"):
                status, body = await request(app, "GET", f"/report.{extension}?v={sample.revision}")
                self.assertEqual(status, 200)
                self.assertIn(b"replayed without running models", body)
                self.assertIn(b"src/t0.h5", body)
            status, body = await request(app, "GET", f"/overlay.png?v={sample.revision}")
            self.assertEqual(status, 200)
            self.assertEqual(body, (sample.directory / "overlay.png").read_bytes())
            status, _ = await request(app, "GET", "/t0.png?v=sample-deadbeefdeadbeef")
            self.assertEqual(status, 410)
            status, _ = await request(app, "GET", f"/unknown.png?v={sample.revision}")
            self.assertEqual(status, 404)

    async def test_sample_playback_does_not_replace_live_pair_or_result(self):
        with tempfile.TemporaryDirectory() as root:
            sample = sample_fixture(root)
            import threading
            demo = SimpleNamespace(lock=threading.Lock(), info={"revision": 42}, result_revision=43,
                                   last_result={"live": True, "evidence_report": {}})
            app = create_app(demo, FRONTEND_ROOT, NAMES, sample=sample)
            status, body = await request(app, "GET", f"/result?v={sample.revision}")
            self.assertEqual(status, 200)
            self.assertIn("sample", json.loads(body)["result"])
            status, body = await request(app, "GET", "/result")
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(body)["result"]["live"])
            self.assertEqual(demo.info, {"revision": 42})
            self.assertEqual(demo.result_revision, 43)
