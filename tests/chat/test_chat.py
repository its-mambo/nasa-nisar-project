import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from nisar.app.paths import FRONTEND_ROOT
from nisar.app.server import create_app
from nisar.features.analysis.service import NAMES
from nisar.features.chat.context import mask_preview
from tests.support import fixture_demo, request


class MaskTests(unittest.TestCase):
    def test_sparse_flags_survive_visual_pooling_and_invalid_cells_are_gray(self):
        valid = np.zeros((4, 8), dtype=bool)
        valid[:, :4] = True
        flags = np.zeros_like(valid)
        flags[3, 3] = True
        flags[3, 7] = True
        self.assertEqual(np.asarray(mask_preview(flags, valid, edge=2)).tolist(), [[255, 127]])


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_chat_streams_and_hands_the_server_snapshot_to_the_engine(self):
        class Engine:
            def __init__(self):
                self.snapshots = []

            def turn(self, revision, client, question, context):
                self.snapshots.append(context())
                self.question = question
                yield {"delta": "Radar "}
                yield {"delta": "evidence."}
                yield {"done": True}

        with TemporaryDirectory() as tmp:
            demo = fixture_demo(Path(tmp))
            demo.infer([0, 0, 64, 64], demo.info["revision"], .1)
            engine = Engine()
            app = create_app(demo, FRONTEND_ROOT, NAMES, chat=engine)
            body = dict(revision=str(demo.result_revision), client="a" * 32, question=" Where? ")
            status, _ = await request(app, "POST", "/chat", body, authorized=False)
            self.assertEqual(status, 403)
            status, _ = await request(app, "POST", "/chat", {**body, "revision": "123456"})
            self.assertEqual(status, 410)
            self.assertEqual(engine.snapshots, [])
            status, data = await request(app, "POST", "/chat", body)
            self.assertEqual(status, 200)
            self.assertEqual([json.loads(line) for line in data.splitlines()],
                             [{"delta": "Radar "}, {"delta": "evidence."}, {"done": True}])
            self.assertEqual(engine.question, "Where?")
            (images, metadata), = engine.snapshots
            self.assertEqual(len(images), 3)
            self.assertNotEqual(images[0].tobytes(), images[1].tobytes())
            self.assertEqual(images[2].getextrema(), ((255, 255),) * 3)
            self.assertEqual(json.loads(metadata)["changed_cells"], 64 ** 2)


    async def test_saved_sample_uses_qwen_and_missing_qwen_reports_the_reason(self):
        from nisar.features.samples.store import SampleSession
        from tests.samples.test_samples import sample_fixture

        class Engine:
            def turn(self, revision, client, question, context):
                self.question = question
                self.images, self.evidence = context()
                yield {"delta": "I see a change."}
                yield {"done": True}

        with TemporaryDirectory() as tmp:
            sample = sample_fixture(tmp)
            demo = SampleSession(sample)
            body = dict(revision=sample.revision, client="b" * 32, question="what has changed?")
            engine = Engine()
            app = create_app(demo, FRONTEND_ROOT, NAMES, sample=sample, model_enabled=False, chat=engine)
            status, data = await request(app, "POST", "/chat", body)
            self.assertEqual(status, 200)
            self.assertEqual(engine.question, body["question"])
            self.assertEqual(len(engine.images), 3)
            self.assertEqual(json.loads(engine.evidence)["mask_source"], "saved_overlay_footprint")
            self.assertEqual(json.loads(data.splitlines()[0]), {"delta": "I see a change."})

            app = create_app(demo, FRONTEND_ROOT, NAMES, sample=sample, model_enabled=False,
                             chat_unavailable="Weights were not found.")
            status, data = await request(app, "POST", "/chat", body)
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(data), {"error": "Weights were not found."})


class CacheTests(unittest.TestCase):
    def test_multimodal_cached_continuation_matches_full_prompt(self):
        from importlib.util import find_spec
        if not all(find_spec(name) for name in ("torch", "transformers", "torchvision")):
            self.skipTest("Optional chat dependencies are not installed.")
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from nisar.features.chat.qwen import QwenChat

        config = Qwen3VLConfig(
            vision_start_token_id=3, vision_end_token_id=4, image_token_id=5, video_token_id=6,
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                               patch_size=2, temporal_patch_size=2, num_position_embeddings=16,
                               out_hidden_size=32, deepstack_visual_indexes=[0]),
        )
        qwen = QwenChat("unused", "unused", torch.device("cpu"))
        qwen.model, qwen.torch = Qwen3VLForConditionalGeneration(config).eval(), torch
        prompt = torch.tensor([[8, 3, 5, 5, 5, 5, 4, 9]])
        visual = {"pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
        suffix = torch.tensor([[10, 11, 12]])
        first = qwen.step(prompt, **visual)
        self.assertNotEqual(first.rope_deltas.item(), 0)
        qwen.step(torch.tensor([[1, 2, 9]]))  # Another client's prefill changes model-global RoPE state.
        cached = qwen.step(suffix, first.past_key_values, first.rope_deltas)
        fresh = qwen.step(torch.cat([prompt, suffix], dim=1), **visual)
        torch.testing.assert_close(cached.logits, fresh.logits, rtol=1e-4, atol=1e-5)
        self.assertEqual(cached.past_key_values.get_seq_length(), prompt.shape[-1] + suffix.shape[-1])
