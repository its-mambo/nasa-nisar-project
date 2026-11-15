"""Verify the packed-file migration preserves only detector bytes and settings."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from nisar.infrastructure.models.detector_weights import FORMAT, extract_detector, read_header


def packed_fixture(path):
    tensors = [
        ("part0.embed", 90),
        ("part1.detector.model.encoder.patch_embed.weight", 10),
        ("part1.linear.model.layers.0.weight", 99),
        ("part1.detector.model.fuse.0.weight", 20),
        ("part1.detector.model.local.0.weight", 30),
        ("part1.detector.model.refine.0.weight", 40),
        ("part2.adapter.weight", 80),
    ]
    detector = {"format": "nisar_change_v1", "size": 224, "stats": {"db_clip": [-50, 10]}}
    metadata = {"format": "nisar_sar_safetensors_v1", "config": json.dumps(
        {"detector": detector, "detector_sha256": "original-digest", "linear": {"private": 1}})}
    bundle = {"parts": [
        {"name": "llm/model.safetensors", "offset": 0, "size": 1},
        {"name": "sar.safetensors", "offset": 1, "size": 5, "metadata": metadata},
        {"name": "adapter/adapter_model.safetensors", "offset": 6, "size": 1},
    ]}
    header = {"__metadata__": {"format": "nisar_single_safetensors_v1", "bundle": json.dumps(bundle)}}
    for i, (name, _) in enumerate(tensors):
        header[name] = {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(v for _, v in tensors))
    return detector


class ExtractionTests(unittest.TestCase):
    def test_extraction_keeps_exact_detector_tensors_and_leaves_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.safetensors"
            output = Path(tmp) / "models" / "detector.safetensors"
            detector = packed_fixture(source)
            original = source.read_bytes()
            extract_detector(source, output)
            self.assertEqual(source.read_bytes(), original)
            with output.open("rb") as stream:
                header, data_offset = read_header(stream)
                metadata = header.pop("__metadata__")
                self.assertEqual(metadata["format"], FORMAT)
                self.assertEqual(json.loads(metadata["config"]), {"detector": detector,
                                                                 "detector_sha256": "original-digest"})
                self.assertEqual(set(header), {"detector.model.encoder.patch_embed.weight",
                    "detector.model.fuse.0.weight", "detector.model.local.0.weight",
                    "detector.model.refine.0.weight"})
                stream.seek(data_offset)
                self.assertEqual(stream.read(), bytes([10, 20, 30, 40]))
            with self.assertRaises(FileExistsError):
                extract_detector(source, output)

    def test_truncated_source_does_not_create_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.safetensors"
            output = Path(tmp) / "detector.safetensors"
            packed_fixture(source)
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "size does not match"):
                extract_detector(source, output)
            self.assertFalse(output.exists())
