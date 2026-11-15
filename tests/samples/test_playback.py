"""Real bundled playback; runs with requirements-sample.txt alone."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen

from nisar.app.paths import PROJECT_ROOT, SAMPLE_ROOT
from nisar.app.web import PAGES, STATIC_FILES


class BundledPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The launcher must find the committed bundle even outside the repo cwd.
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.log = tempfile.TemporaryFile()
        cls.addClassCleanup(cls.log.close)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "launch.py"), "--mode", "sample",
             "--no-chat", "--port", str(port)], cwd=cls.work.name,
            stdout=cls.log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                break
            try:
                with urlopen(cls.base + "/sample", timeout=1) as response:
                    cls.capabilities = json.load(response)
                return
            except (URLError, TimeoutError):
                time.sleep(.1)
        cls.log.seek(0)
        raise AssertionError("Sample server did not start:\n" + cls.log.read().decode("utf-8", errors="replace"))

    @classmethod
    def stop_server(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=10)

    def fetch(self, path):
        return urlopen(self.base + path, timeout=20)

    def test_every_bundled_image_is_served_with_its_original_checksum(self):
        self.assertFalse(self.capabilities["model_enabled"])
        sample = self.capabilities["sample"]
        self.assertTrue(sample["available"])
        revision = sample["revision"]
        manifest = json.loads((SAMPLE_ROOT / revision / "manifest.json").read_text(encoding="utf-8"))
        with self.fetch("/result?v=" + revision) as response:
            result = json.load(response)["result"]
        self.assertEqual(result["sample"]["revision"], revision)
        self.assertEqual(result["acquisitions"]["t0"]["zeroDopplerStartTime"][:10], "2026-06-21")
        self.assertEqual(result["acquisitions"]["t1"]["zeroDopplerStartTime"][:10], "2026-09-13")
        images = {**result["images"], "sample-preview.png": sample["preview_url"]}
        for name, url in images.items():
            with self.subTest(image=name), self.fetch(url) as response:
                self.assertEqual(response.headers.get_content_type(), "image/png")
                self.assertEqual(hashlib.file_digest(response, "sha256").hexdigest(), manifest["files"][name])

    def test_pages_scripts_and_report_are_available_without_models(self):
        for path in (*PAGES, *("/" + name for name in STATIC_FILES)):
            with self.subTest(path=path), self.fetch(path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        with self.fetch("/report.md?v=" + self.capabilities["sample"]["revision"]) as response:
            self.assertIn("detector", response.read().decode("utf-8").lower())

    def test_sample_json_loading_is_independent_of_default_text_encoding(self):
        # Turn implicit locale-dependent text reads into errors on every OS.
        # The real result and regional context both contain non-ASCII text.
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'h5py', 'transformers', 'PIL', 'safetensors'}:
        raise AssertionError('Sample playback imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from nisar.app.paths import SAMPLE_ROOT
from nisar.features.samples.store import CachedSample, SampleSession
sample = CachedSample(SAMPLE_ROOT)
result = sample.result()
assert result['evidence_report']['external_context']
SampleSession(sample)
"""
        subprocess.run([sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-c", script],
                       cwd=PROJECT_ROOT, check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
