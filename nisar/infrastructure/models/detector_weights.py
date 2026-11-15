"""Extract the detector state directly from an existing packed checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import struct

FORMAT = "nisar_detector_safetensors_v1"
PREFIX = "detector.model."
GROUPS = ("encoder.", "fuse.", "local.", "refine.")


def read_header(stream):
    size = stream.read(8)
    if len(size) != 8:
        raise ValueError("Truncated safetensors header.")
    length, = struct.unpack("<Q", size)
    if not 2 <= length <= 100_000_000:
        raise ValueError("Invalid safetensors header size.")
    raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise ValueError("Invalid safetensors header.")
    header = json.loads(raw)
    end = stream.seek(0, 2)
    start = 8 + length
    cursor = 0
    for lo, hi in sorted(x["data_offsets"] for k, x in header.items() if k != "__metadata__"):
        if type(lo) is not int or type(hi) is not int or lo != cursor or hi < lo:
            raise ValueError("Invalid tensor offsets.")
        cursor = hi
    if start + cursor != end:
        raise ValueError("Safetensors size does not match its tensor offsets.")
    return header, start


def extract_detector(source, output):
    if output.suffix != ".safetensors":
        raise ValueError("Output must end in .safetensors.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    with source.open("rb") as src:
        header, start = read_header(src)
        metadata = header.pop("__metadata__", {})
        offset, size, part_prefix = 0, source.stat().st_size - start, ""
        if metadata.get("format") == "nisar_single_safetensors_v1":
            parts = json.loads(metadata["bundle"])["parts"]
            matches = [(i, p) for i, p in enumerate(parts) if p["name"] == "sar.safetensors"]
            if len(matches) != 1:
                raise ValueError("Expected one SAR component in the packed checkpoint.")
            index, part = matches[0]
            offset, size = part["offset"], part["size"]
            if (type(offset) is not int or type(size) is not int or offset < 0 or size <= 0
                    or start + offset + size > source.stat().st_size):
                raise ValueError("Invalid SAR component range.")
            metadata, part_prefix = part["metadata"], f"part{index}."
        if not isinstance(metadata, dict) or metadata.get("format") != "nisar_sar_safetensors_v1":
            raise ValueError("Expected the original packed NISAR checkpoint or its SAR component.")
        config = json.loads(metadata["config"])
        detector = config["detector"]
        if detector.get("format") != "nisar_change_v1" or "model" in detector:
            raise ValueError("Expected NISAR change detector configuration.")
        prefix = part_prefix + PREFIX
        selected = [(key[len(prefix):], spec) for key, spec in header.items() if key.startswith(prefix)]
        if any(not any(name.startswith(g) for g in GROUPS) for name, _ in selected) or any(
                not any(name.startswith(g) for name, _ in selected) for g in GROUPS):
            raise ValueError("Expected the encoder and all three detector head branches.")
        new_header = {"__metadata__": {"format": FORMAT, "config": json.dumps(
            {"detector": detector, "detector_sha256": config.get("detector_sha256")}, allow_nan=False)}}
        cursor = 0
        spans = []
        for name, spec in sorted(selected, key=lambda pair: pair[1]["data_offsets"]):
            lo, hi = spec["data_offsets"]
            if not offset <= lo <= hi <= offset + size:
                raise ValueError("Detector tensor falls outside its SAR component.")
            new_header[PREFIX + name] = {**spec, "data_offsets": [cursor, cursor + hi - lo]}
            spans.append((lo, hi-lo))
            cursor += hi-lo
        encoded = json.dumps(new_header, separators=(",", ":"), allow_nan=False).encode()
        encoded += b" " * (-len(encoded) % 8)
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < cursor + len(encoded) + 8:
            raise OSError("Not enough free disk space for the detector.")
        with output.open("xb") as dst:
            try:
                dst.write(struct.pack("<Q", len(encoded)))
                dst.write(encoded)
                for lo, remaining in spans:
                    src.seek(start+lo)
                    while remaining:
                        chunk = src.read(min(remaining, 8*1024**2))
                        if not chunk:
                            raise ValueError("Truncated detector tensor.")
                        dst.write(chunk)
                        remaining -= len(chunk)
            except BaseException:
                dst.close()
                output.unlink()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        extract_detector(args.weights, args.out)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Extraction failed: {exc}\n")
    print(f"Saved {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
