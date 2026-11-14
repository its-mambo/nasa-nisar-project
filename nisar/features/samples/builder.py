"""Run the real model once and publish a portable, checksummed sample bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from nisar.app.paths import PROJECT_ROOT, SAMPLE_ROOT
from nisar.features.analysis.artifacts import NAMES
from nisar.features.samples.spec import ROI, SOURCE_FILES, THRESHOLD
from nisar.features.samples.store import CachedSample, FORMAT, digest, load_sample


def publish(export, source_dir, destination):
    from PIL import Image
    import h5py
    from nisar.features.imagery.gcov import metadata

    export, source_dir, destination = map(Path, (export, source_dir, destination))
    result = json.loads((export / "report.json").read_text())
    if result["roi"]["t0_pixel_box"] != ROI or result["threshold"] != THRESHOLD:
        raise ValueError("Export must use the fixed sample ROI and threshold 0.10.")
    sources = {}
    for date, name in SOURCE_FILES.items():
        path = source_dir / name
        with h5py.File(path, "r") as product:
            acquisition = metadata(product)
        if acquisition != result["acquisitions"][date]:
            raise ValueError(f"{date.upper()} source metadata differs from the exported run.")
        sources[date] = {"name": name, "path": f"src/{name}", "bytes": path.stat().st_size,
                         "sha256": digest(path), "acquisition": acquisition}
    sample = dict(title="Saved analysis sample", roi=ROI, threshold=THRESHOLD, sources=sources,
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  native_size_px=[ROI[2]-ROI[0], ROI[3]-ROI[1]],
                  projected_area_km2=result["roi"]["projected_rectangle_km2"],
                  epsg=result["roi"]["epsg"],
                  detector_sha256=result["analysis_settings"].get("detector_sha256"),
                  interpretation_source=result["interpretation"]["source"])
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=destination) as work:
        staging = Path(work)
        for name in NAMES:
            with Image.open(export / name) as image:
                if list(image.size) != sample["native_size_px"]:
                    raise ValueError(f"Incorrect sample image dimensions: {name}")
                image.verify()
            shutil.copyfile(export / name, staging / name)
        with Image.open(export / "t0.png") as preview:
            preview.thumbnail((1200, 800))
            preview.save(staging / "sample-preview.png")
        (staging / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        files = {name: digest(staging / name) for name in (*NAMES, "sample-preview.png", "result.json")}
        revision = "sample-" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
        manifest = dict(format=FORMAT, revision=revision, sample=sample, files=files)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        final = destination / revision
        if not final.exists():
            staging.rename(final)
        pointer = destination / ".current.json.tmp"
        pointer.write_text(json.dumps({"revision": revision}) + "\n")
        pointer.replace(destination / "current.json")
    return CachedSample(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "src")
    parser.add_argument("--cache-dir", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--from-export", type=Path, help="Import a completed real one-shot run instead of rerunning models.")
    parser.add_argument("--refresh", action="store_true", help="Generate another run; keep the previous immutable bundle.")
    args, model_args = parser.parse_known_args()
    existing = load_sample(args.cache_dir)
    if existing and not args.refresh:
        print(f"Reusing {existing.revision}; use --refresh to run again.")
        return
    if args.from_export:
        if model_args:
            parser.error("Model options cannot be combined with --from-export.")
        sample = publish(args.from_export, args.source_dir, args.cache_dir)
    else:
        reserved = {"--roi", "--threshold", "--t0", "--t1", "--output-dir"}
        if any(option.split("=")[0] in reserved for option in model_args):
            parser.error("The sample's input files, ROI, threshold, and export location are fixed.")
        with tempfile.TemporaryDirectory(prefix="nisar-sample-build-") as work:
            export = Path(work) / "analysis"
            command = [sys.executable, str(PROJECT_ROOT / "inference.py"), *model_args,
                       "--t0", str(args.source_dir / SOURCE_FILES["t0"]),
                       "--t1", str(args.source_dir / SOURCE_FILES["t1"]),
                       "--roi", *map(str, ROI), "--threshold", str(THRESHOLD), "--output-dir", str(export)]
            completed = subprocess.run(command, cwd=PROJECT_ROOT)
            # Exit 1 can mean Qwen failed after a successful detector run and export.
            if completed.returncode not in (0, 1) or not (export / "report.json").is_file():
                parser.exit(1, "Sample generation failed; the previous cache is unchanged.\n")
            sample = publish(export, args.source_dir, args.cache_dir)
    print(f"Saved {sample.revision} to {sample.directory}")
    print("Judges can now run: python launch.py --mode sample")


if __name__ == "__main__":
    main()
