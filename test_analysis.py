"""Regression checks without loading the bundled model weights.

Run with: python -m unittest -v test_analysis
"""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.parse import urlsplit

import h5py
import numpy as np
from PIL import Image
import torch

from change_model import infer_scene
from inference import Demo, NAMES, ROOT
from server import create_app


class ConstantDetector(torch.nn.Module):
    def forward(self, a, b):
        return torch.full((len(a), 224, 224), torch.logit(torch.tensor(.6)))


class DetectorProgressTests(unittest.TestCase):
    def test_threshold_changes_mask_and_reports_all_tiles(self):
        def pair(box):
            x, y, right, bottom = box
            db = np.zeros((2, bottom-y, right-x), np.float32)
            # A whole tile is invalid; it still counts toward work completed.
            valid = np.full((bottom-y, right-x), not (x == 224 and y == 224))
            return (db, valid), (db, valid)

        for threshold, expected in ((.1, 300*250-76*26), (.9, 0)):
            with self.subTest(threshold=threshold):
                updates = []
                mask, _, valid, changed = infer_scene(
                    ConstantDetector(), (300, 250), pair, np.zeros((2,1,1)),
                    np.ones((2,1,1)), np.array([[-30,10],[-30,10]]),
                    torch.device("cpu"), "off", 2, 0, threshold,
                    progress=lambda done, total: updates.append((done, total)))
                self.assertEqual(valid, 300*250-76*26)
                self.assertEqual(changed, expected)
                self.assertEqual(int(mask.sum()), expected)
                self.assertEqual(updates, [(0, 4), (2, 4), (4, 4)])


class FakeSar:
    spec = {"frequency":"A", "spacing_m":10, "db_clip":[-30, 5]}
    device = SimpleNamespace(type="cpu")

    def detect(self, size, pair, *, threshold, progress, **kwargs):
        a, b = pair((0, 0, *size))
        valid = a[1] & b[1]
        mask = valid & (.6 >= threshold)
        progress(0, 1)
        progress(1, 1)
        return mask, None, int(valid.sum()), int(mask.sum())

    def classify(self, size, pair, mask):
        return {"status":"insufficient_candidates", "predictions":{}, "reason":"Fixture selection."}


class FakeQwen:
    device = SimpleNamespace(type="cpu")
    has_adapter = False

    def report(self, *args):
        return "Test explanation based on measured backscatter."


def fixture_demo(directory):
    options = SimpleNamespace(threshold=.1, max_roi_pixels=1_000_000,
        batch_size=2, overlap=0, decoder_overview_size=256, serial_vision=True,
        inspect_decoder_tensors=False, max_delta_db=8, max_upload_gib=1)
    demo = Demo(directory, options, FakeSar(), FakeQwen())
    size = 128
    yy, xx = np.mgrid[:size, :size]
    for date, scale, stamp in (("t0", 1., "2026-06-21T00:00:00Z"),
                                ("t1", 2., "2026-09-13T00:00:00Z")):
        with h5py.File(Path(directory) / f"{date}.h5", "w") as product:
            grid = product.create_group("/science/LSAR/GCOV/grids/frequencyA")
            grid["HHHH"] = ((.01 + xx / 5000 + yy / 10000) * scale).astype(np.float32)
            grid["HVHV"] = ((.002 + yy / 15000) * scale).astype(np.float32)
            grid["xCoordinateSpacing"], grid["yCoordinateSpacing"] = 10., -10.
            grid["xCoordinates"] = 500005. + np.arange(size)*10
            grid["yCoordinates"] = 4500005. - np.arange(size)*10
            grid["projection"] = 32640
            ident = product.create_group("/science/LSAR/identification")
            ident["zeroDopplerStartTime"] = stamp
    demo.prepare()
    return demo


class InferenceOutputTests(unittest.TestCase):
    def test_per_run_threshold_and_independent_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = fixture_demo(directory)
            stages = []
            result = demo.infer([0,0,128,128], demo.info["revision"], .1,
                                lambda stage, *args: stages.append(stage))
            self.assertEqual(result["threshold"], .1)
            self.assertEqual(result["changed_pixels"], 128**2)
            self.assertEqual(list(dict.fromkeys(stages)), ["read","detect","classify","report","render"])
            self.assertIn("overlay.png", result["images"])
            self.assertTrue(result["viewer_url"].startswith("/change?v="))
            overlay = Image.open(Path(directory) / "overlay.png").convert("RGBA")
            self.assertTrue(np.any(np.asarray(overlay)[:,:,3] > 0))
            # Both precomposited downloads equal the independent overlay on their base.
            for base, combined in (("t0", "change"), ("t0_hh", "change_hh")):
                expected = Image.alpha_composite(Image.open(Path(directory)/f"{base}.png"), overlay)
                np.testing.assert_array_equal(np.asarray(expected), np.asarray(Image.open(Path(directory)/f"{combined}.png")))
            hh = np.asarray(Image.open(Path(directory) / "t0_hh.png"))
            np.testing.assert_array_equal(hh[:,:,0], hh[:,:,1])
            np.testing.assert_array_equal(hh[:,:,1], hh[:,:,2])

            result = demo.infer([0,0,128,128], demo.info["revision"], .9)
            self.assertEqual(result["changed_pixels"], 0)
            self.assertEqual(demo.options.threshold, .1)
            self.assertFalse(np.asarray(Image.open(Path(directory)/"overlay.png"))[:,:,3].any())
            # Omitting a per-run value continues to respect the command-line default.
            self.assertEqual(demo.infer([0,0,128,128], demo.info["revision"])["threshold"], .1)
            for invalid in (0, 1, -.1, "0.5", True, float("nan"), float("inf")):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    demo.infer([0,0,128,128], demo.info["revision"], invalid)


async def request(app, method, path, body=None, authorized=True):
    """Exercise ASGI directly, without a network listener or httpx dependency."""
    parsed = urlsplit(path)
    payload = b"" if body is None else json.dumps(body).encode()
    sent, delivered = [], False
    headers = [(b"content-type", b"application/json")]
    if authorized:
        headers.append((b"x-nisar-demo", b"1"))
    scope = dict(type="http", asgi={"version":"3.0", "spec_version":"2.4"},
        method=method, path=parsed.path, raw_path=parsed.path.encode(),
        query_string=parsed.query.encode(), headers=headers, http_version="1.1",
        scheme="http", server=("test",80), client=("test",123))

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type":"http.request", "body":payload, "more_body":False}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), 3)
    status = next(item["status"] for item in sent if item["type"] == "http.response.start")
    data = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
    return status, data


class JobApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonblocking_progress_conflict_result_and_errors(self):
        entered, release = threading.Event(), threading.Event()
        class JobDemo:
            lock = threading.Lock()

            def infer(self, box, revision, threshold, progress):
                progress("detect", "Processed 2 of 5 image tiles", 2, 5)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test timed out")
                if threshold == .99:
                    raise ValueError("Test failure")
                return {"threshold":threshold, "changed_pixels":10}

        app = create_app(JobDemo(), ROOT, NAMES)
        status, _ = await request(app, "POST", "/infer", {}, authorized=False)
        self.assertEqual(status, 403)
        status, _ = await request(app, "POST", "/infer", [])
        self.assertEqual(status, 400)
        status, raw = await request(app, "POST", "/infer", {"threshold":.5})
        self.assertEqual(status, 202)
        run_id = json.loads(raw)["run_id"]
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            # The model lock is held by the worker. Progress still responds promptly.
            status, raw = await asyncio.wait_for(request(app, "GET", f"/infer/{run_id}"), 1)
            job = json.loads(raw)
            self.assertEqual(status, 200)
            self.assertEqual((job["stage"],job["completed"],job["total"]), ("detect",2,5))
            status, _ = await request(app, "POST", "/infer", {"threshold":.5})
            self.assertEqual(status, 409)
        finally:
            release.set()
        for _ in range(100):
            _, raw = await request(app, "GET", f"/infer/{run_id}")
            job = json.loads(raw)
            if job["status"] != "running":
                break
            await asyncio.sleep(.01)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"]["threshold"], .5)
        status, raw = await request(app, "POST", "/infer", {"threshold":.99})
        failed_id = json.loads(raw)["run_id"]
        for _ in range(100):
            _, raw = await request(app, "GET", f"/infer/{failed_id}")
            job = json.loads(raw)
            if job["status"] != "running":
                break
            await asyncio.sleep(.01)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Test failure")
        status, _ = await request(app, "GET", "/infer/missing")
        self.assertEqual(status, 404)
        status, _ = await request(app, "GET", "/navigation.css")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
