"""Regression checks without loading the bundled model weights.

Run with: python -m unittest discover -s tests -t . -v
"""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest

import h5py
import numpy as np
from PIL import Image
from importlib.util import find_spec
from nisar.features.analysis.service import NAMES
from nisar.features.imagery.rendering import aligned_power
from nisar.app.paths import FRONTEND_ROOT
from tests.support import fixture_demo, request
from nisar.app.server import create_app
from nisar.features.analysis.progress_timing import RunTiming


class RemainingTimeTests(unittest.TestCase):
    def test_estimates_use_actual_stage_work_and_do_not_invent_future_times(self):
        now = [0.]
        timing = RunTiming(clock=lambda:now[0])
        timing.update("detect",0,10)
        self.assertIsNone(timing.estimate())
        now[0] = 5
        timing.update("detect",2,10)
        self.assertEqual(timing.estimate(),dict(scope="stage",basis="work completed in this stage",lower_s=12,upper_s=32))
        timing.update("measure")
        self.assertIsNone(timing.estimate())
        timing.update("render",0,10)
        now[0] = 9
        timing.update("render",2,10)
        self.assertEqual(timing.estimate()["scope"],"analysis")
        self.assertEqual(timing.profile()["durations"]["render"],4)

    def test_history_scaling_and_unrelated_or_overdue_runs(self):
        history = [{"pixels":100,"durations":dict(read=2,detect=10,measure=7,render=6)}]
        now = [0.]
        timing = RunTiming(pixels=200,history=history,clock=lambda:now[0])
        timing.update("read")
        now[0] = 1
        estimate = timing.estimate()
        self.assertEqual((estimate["scope"],estimate["lower_s"],estimate["upper_s"]),("analysis",29,78))
        now[0] = 10
        self.assertIsNone(timing.estimate())
        timing = RunTiming(pixels=10000,history=history,clock=lambda:now[0])
        timing.update("render")
        self.assertIsNone(timing.estimate())


@unittest.skipUnless(find_spec("torch"), "PyTorch is required for detector inference")
class DetectorProgressTests(unittest.TestCase):
    def test_threshold_changes_mask_and_reports_all_tiles(self):
        import torch
        from nisar.infrastructure.models.change_model import infer_scene

        class ConstantDetector(torch.nn.Module):
            def forward(self, a, b):
                return torch.full((len(a), 224, 224), torch.logit(torch.tensor(.6)))

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


class InferenceOutputTests(unittest.TestCase):
    def test_later_preview_is_aligned_and_tiles_match(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = fixture_demo(directory)
            before = np.asarray(Image.open(Path(directory)/"preview.png"))
            after = np.asarray(Image.open(Path(directory)/"preview_t1.png"))
            self.assertEqual(before.shape,after.shape)
            self.assertFalse(np.array_equal(before,after))
            # A nonzero product offset must not move the selection grid.
            with h5py.File(Path(directory)/"t1.h5","r+") as product:
                grid = product["/science/LSAR/GCOV/grids/frequencyA"]
                grid["xCoordinates"][:] += 30
                grid["yCoordinates"][:] += 20
            demo.prepare()
            self.assertEqual(demo.info["t1_shift"],(-3,2))
            for mode, filename in (("rg","preview_t1.png"),("hh","preview_t1_hh.png")):
                later = np.asarray(Image.open(Path(directory)/filename))
                np.testing.assert_array_equal(later,np.asarray(demo.preview_tile(mode,1,0,0,date="t1")))
                self.assertFalse(later[:,:3,3].any())
                self.assertFalse(later[-2:,:,3].any())
                self.assertTrue((later[:-2,3:,3] == 255).all())
            with self.assertRaises(ValueError):
                demo.preview_tile("rg",1,0,0,date="other")

    def test_shifted_coarse_preview_uses_the_same_sampling_lattice(self):
        class SampleGrid:
            width, height = 11, 9
            def power(self,x,y,right,bottom,step):
                yy,xx = np.mgrid[y:bottom:step,x:right:step]
                data = (100*yy+xx+1).astype(np.float32)
                return np.stack([data,data]), np.ones(data.shape,bool)
        # Offset not divisible by sample spacing, including an interior tile origin.
        for shift in ((-3,2),(2,-3),(-50,0),(0,50)):
            for box in ((0,0,16,12),(5,1,13,10)):
                power,valid = aligned_power(SampleGrid(),box,shift,4)
                for row,y in enumerate(range(box[1],box[3],4)):
                    for col,x in enumerate(range(box[0],box[2],4)):
                        bx,by = x+shift[0], y+shift[1]
                        inside = 0 <= bx < 11 and 0 <= by < 9
                        self.assertEqual(bool(valid[row,col]),inside)
                        self.assertEqual(power[0,row,col],100*by+bx+1 if inside else 0)

    def test_per_run_threshold_and_independent_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = fixture_demo(directory)
            stages = []
            result = demo.infer([0,0,128,128], demo.info["revision"], .1,
                                lambda stage, *args: stages.append(stage))
            self.assertEqual(result["threshold"], .1)
            self.assertEqual(result["changed_pixels"], 128**2)
            self.assertEqual(list(dict.fromkeys(stages)), ["read","detect","measure","render"])
            self.assertIn("overlay.png", result["images"])
            self.assertEqual(result["analysis_settings"]["channels"],["HH","HV"])
            self.assertEqual(result["analysis_settings"]["detector_overlap_px"],0)
            self.assertTrue(result["viewer_url"].startswith("/change?v="))
            overlay = Image.open(Path(directory) / "overlay.png").convert("RGBA")
            self.assertTrue(np.any(np.asarray(overlay)[:,:,3] > 0))
            # Both precomposited downloads equal the independent overlay on their base.
            for base, combined in (("t0", "change"), ("t1", "change_t1"),
                                   ("t0_hh", "change_hh"), ("t1_hh", "change_t1_hh")):
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

        app = create_app(JobDemo(), FRONTEND_ROOT, NAMES)
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
