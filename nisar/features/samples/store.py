"""Read an immutable analysis bundle using only the Python standard library."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading

from nisar.features.analysis.artifacts import NAMES
from nisar.features.samples.context import regional_context

REVISION = re.compile(r"sample-[0-9a-f]{16}")
FORMAT = "nisar_cached_sample_v1"
IMAGE_FILES = frozenset((*NAMES, "sample-preview.png"))


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class CachedSample:
    def __init__(self, root):
        root = Path(root)
        self.revision = json.loads((root / "current.json").read_text())["revision"]
        if not isinstance(self.revision, str) or not REVISION.fullmatch(self.revision):
            raise ValueError("Invalid cached sample revision.")
        self.directory = root / self.revision
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest.get("format") != FORMAT or manifest.get("revision") != self.revision:
            raise ValueError("Unsupported or mismatched sample manifest.")
        files = manifest["files"]
        if set(files) != IMAGE_FILES | {"result.json"}:
            raise ValueError("Cached sample is missing required artifacts.")
        for name, checksum in files.items():
            if digest(self.directory / name) != checksum:
                raise ValueError(f"Cached sample checksum mismatch: {name}")
        self.metadata = manifest["sample"]
        self._result = json.loads((self.directory / "result.json").read_text())
        if (self._result["roi"]["t0_pixel_box"] != self.metadata["roi"]
                or self._result["threshold"] != self.metadata["threshold"]):
            raise ValueError("Cached result does not match its sample settings.")

    def describe(self):
        return {**deepcopy(self.metadata), "available": True, "revision": self.revision,
                "preview_url": f"/sample-preview.png?v={self.revision}"}

    def result(self):
        result = deepcopy(self._result)
        result["sample"] = self.describe()
        result["images"] = {name: f"/{name}?v={self.revision}" for name in NAMES}
        result["viewer_url"] = f"/change?v={self.revision}"
        # The portable cache contains the result and images, not live decoder tensors.
        result.pop("decoder_inputs_url", None)
        context = regional_context(result)
        if context and isinstance(result.get("evidence_report"), dict):
            report = result["evidence_report"]
            report["external_context"] = context
            report["validation"] = report.get("validation", "").replace(
                "No external sources were consulted.", "").strip() + (
                " The original model run did not consult external sources. Regional context was reviewed "
                "separately and does not validate detector accuracy or physical attribution.")
        return result


class SampleSession:
    """Read-only server state; never imports a model, NumPy, or an HDF5 reader."""
    def __init__(self, sample):
        self.lock = threading.Lock()
        self.info = None
        self.dir = sample.directory
        self.result_revision = sample.revision
        self.last_result = sample.result()
        self.decoder_trace = None


def load_sample(root):
    if not (Path(root) / "current.json").is_file():
        return None
    return CachedSample(root)
