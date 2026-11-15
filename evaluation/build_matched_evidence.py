"""Build a portable, location-matched pilot from original radar and Landsat data."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nisar.features.imagery.gcov import Grid, db, metadata
from nisar.features.samples.store import CachedSample, digest
from fetch_landsat import BANDS, ITEMS, STAC

SCALE, OFFSET = .0000275, -.2
INDEX_THRESHOLD = .15


def optical_metrics(raw):
    """Compute reference diagnostics on original 30 m cells, never upsampled cells."""
    quality = raw[5]
    good = ((quality & 63) == 0) & (raw[6] == 0) & (raw[:5] > 0).all(0)
    rho = raw[:5].astype(np.float64) * SCALE + OFFSET
    denominator = rho[1] + rho[4]
    usable_index = good & (denominator > 1e-6)
    mndwi = np.divide(rho[1] - rho[4], denominator,
                      out=np.zeros_like(denominator), where=denominator > 1e-6)
    n = int(good.sum())
    index_n = int(usable_index.sum())
    return dict(total_cells=int(good.size), usable_cells=n,
                usable_fraction=n / good.size,
                mndwi_usable_cells=index_n,
                median_mndwi=float(np.median(mndwi[usable_index])) if index_n else None,
                water_like_cells=int(((mndwi > INDEX_THRESHOLD) & usable_index).sum()),
                water_like_fraction=float(((mndwi > INDEX_THRESHOLD) & usable_index).sum() / index_n) if index_n else None,
                source_water_flag_cells=int((good & ((quality & 128) != 0)).sum()),
                source_water_flag_fraction=int((good & ((quality & 128) != 0)).sum()) / n if n else None)


def radar_metrics(z0, z1, valid, alpha):
    return dict(paired_cells=int(valid.sum()), total_cells=int(valid.size),
                median_delta_hh_hv_db=np.median((z1 - z0)[:, valid], axis=1).astype(float).tolist(),
                visible_flag_cells=int(((alpha > 0) & valid).sum()),
                visible_flag_fraction=int(((alpha > 0) & valid).sum()) / int(valid.sum()),
                flag_scope="Nonzero alpha in the original saved overlay; lower bound, not the complete detector mask")


def save_display(raw, transform, bounds, path, false_color=False):
    west, south, east, north = bounds
    height, width = round((north - south) / 20), round((east - west) / 20)
    target = np.zeros((7, height, width), dtype=np.uint16)
    for i in range(7):
        reproject(raw[i], target[i], src_transform=transform, src_crs="EPSG:32640",
                  dst_transform=from_origin(west, north, 20, 20), dst_crs="EPSG:32640",
                  resampling=Resampling.nearest)
    valid = ((target[5] & 63) == 0) & (target[6] == 0) & (target[:5] > 0).all(0)
    rho = target[:5].astype(np.float64) * SCALE + OFFSET
    channels = [4, 3, 0] if false_color else [0, 1, 2]
    display = np.moveaxis(np.clip(rho[channels] / .6, 0, 1) ** .7, 0, -1)
    display[~valid] = (.6, .6, .6)
    Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--landsat-cache", type=Path, required=True)
    args = parser.parse_args()
    sample = CachedSample(ROOT / "samples/judges")
    original = sample.result()
    if sample.revision != "sample-0893b8c3af904967":
        parser.error("This pilot is tied to sample-0893b8c3af904967.")
    output = ROOT / "evaluation/karakalpakstan-pilot"
    source_out = output / "source"
    source_out.mkdir(parents=True, exist_ok=True)
    images_out = ROOT / "frontend/features/results/reference/matched"
    images_out.mkdir(parents=True, exist_ok=True)
    sites = json.loads(Path(__file__).with_name("sites.json").read_text(encoding="utf-8"))
    sources = []
    items = {}
    for label, ident in ITEMS.items():
        item = json.loads((args.landsat_cache / f"{label}-item.json").read_text(encoding="utf-8"))
        if item["id"] != ident:
            raise ValueError("Unexpected reference acquisition")
        items[label] = item
        shutil.copyfile(args.landsat_cache / f"{label}-item.json", source_out / f"{label}-item.json")
        day = item["properties"]["datetime"][:10]
        sources.append(dict(id=ident, label=label, sensor=item["properties"]["platform"],
            datetime=item["properties"]["datetime"], date=day, resolution_m=30,
            radar_date=("2026-06-21" if label == "before" else "2026-09-13"),
            offset_days=-3 if label == "before" else 1,
            url=f"{STAC}/collections/landsat-c2-l2/items/{ident}",
            scene_cloud_cover_percent=item["properties"].get("eo:cloud_cover")))
    handles = []
    try:
        for label in ("t0", "t1"):
            record = sample.metadata["sources"][label]
            path = args.source_dir / record["name"]
            if digest(path) != record["sha256"]:
                raise ValueError("Radar source checksum differs from the saved sample: " + label)
            handles.append(h5py.File(path, "r"))
        grids = [Grid.read(h, "A") for h in handles]
        for label, handle, grid in zip(("t0", "t1"), handles, grids):
            if metadata(handle) != original["acquisitions"][label] or grid.epsg != 32640:
                raise ValueError("Radar acquisition or projection mismatch")
            if grid.bounds(sample.metadata["roi"]) != original["roi"]["projected_bounds_wsen"]:
                raise ValueError("Radar coordinates differ from saved result")
        west, south, east, north = original["roi"]["projected_bounds_wsen"]
        project = Transformer.from_crs(32640, 4326, always_xy=True)
        with Image.open(sample.directory / "overlay.png") as im:
            saved_alpha = np.asarray(im.getchannel("A"))
        for site in sites:
            ident = site["id"]
            w, s, e, n = site["bounds_wsen"]
            col, row = round((w - west) / 20), round((north - n) / 20)
            width, height = round((e - w) / 20), round((n - s) / 20)
            box = [col, row, col + width, row + height]
            site["sample_pixel_box"] = box
            site["viewport_box"] = [col / 5932, row / 3652, (col + width) / 5932, (row + height) / 3652]
            lon, lat = project.transform((w + e) / 2, (s + n) / 2)
            site["center_lon_lat"] = [round(lon, 6), round(lat, 6)]
            site["area_km2"] = (e - w) * (n - s) / 1e6
            if "focus_point_en" in site:
                px, py = site["focus_point_en"]
                site["comparison_point"] = dict(easting_m=px, northing_m=py,
                    image_fraction=[(px - w) / (e - w), (n - py) / (n - s)],
                    viewport_point=[(px - west) / 118640, (north - py) / 73040],
                    optical={})
            radar, valid = [], []
            for grid in grids:
                native_box = [sample.metadata["roi"][0] + col, sample.metadata["roi"][1] + row,
                              sample.metadata["roi"][0] + col + width, sample.metadata["roi"][1] + row + height]
                power, good = grid.power(*native_box)
                radar.append(db(power)); valid.append(good)
            paired = valid[0] & valid[1]
            alpha = saved_alpha[row:row + height, col:col + width]
            site["radar"] = radar_metrics(*radar, paired, alpha)
            np.savez_compressed(source_out / f"{ident}-radar.npz", t0_db=radar[0], t1_db=radar[1],
                                paired_valid=paired, overlay_alpha=alpha)
            site["images"] = {}
            for label, name in (("radar_before", "t0.png"), ("radar_after", "t1.png"), ("overlay", "overlay.png")):
                filename = f"{ident}-{label}.png"
                with Image.open(sample.directory / name) as im:
                    im.crop(box).save(images_out / filename)
                site["images"][label] = "/matched/" + filename
            site["optical"] = {}
            site["source_crops"] = {}
            transforms = []
            for label in ("before", "after"):
                raw_bands = []
                for band in BANDS:
                    with rasterio.open(args.landsat_cache / f"{label}-{band}.tif") as src:
                        c = int(np.ceil((w - src.transform.c) / 30 - .5))
                        r = int(np.ceil((src.transform.f - n) / 30 - .5))
                        window = Window(c, r, round((e - w) / 30), round((n - s) / 30))
                        raw_bands.append(src.read(1, window=window))
                        transform = src.window_transform(window)
                raw = np.stack(raw_bands)
                if raw.shape != (7, 100, 100):
                    raise ValueError("Incomplete reference crop")
                transforms.append(transform)
                filename = f"{ident}-{label}.tif"
                with rasterio.open(source_out / filename, "w", driver="GTiff", width=100, height=100,
                                   count=7, dtype="uint16", transform=transform, crs="EPSG:32640", compress="deflate") as dst:
                    dst.write(raw)
                    for i, band in enumerate(BANDS, 1):
                        dst.set_band_description(i, band)
                    dst.update_tags(stac_item=ITEMS[label], reflectance_scale=SCALE, reflectance_offset=OFFSET)
                site["source_crops"][label] = "/matched-source/" + filename
                site["optical"][label] = optical_metrics(raw)
                site["optical"][label]["transform"] = list(transform)[:6]
                if "comparison_point" in site:
                    point = site["comparison_point"]
                    pc, pr = (~transform) * (point["easting_m"], point["northing_m"])
                    pr, pc = int(np.floor(pr)), int(np.floor(pc))
                    point["optical"][label] = dict(native_col=pc, native_row=pr,
                        qa_pixel=int(raw[5, pr, pc]), **optical_metrics(raw[:, pr:pr + 1, pc:pc + 1]))
                for mode in ("natural", "false_color"):
                    filename = f"{ident}-{label}-{mode}.png"
                    save_display(raw, transform, site["bounds_wsen"], images_out / filename, mode == "false_color")
                    site["images"][label + "_" + mode] = "/matched/" + filename
            if transforms[0] != transforms[1]:
                raise ValueError("The two native optical crops must share the same grid")
    finally:
        for handle in handles:
            handle.close()
    result = dict(
        schema_version="nisar_matched_pilot_v1", id="karakalpakstan-optical-pilot-v1",
        sample_revision=sample.revision, generated_at=datetime.now(timezone.utc).isoformat(), epsg=32640,
        title="Same locations, independent optical checks",
        summary="Four selected 3 km squares compare the saved radar result with georeferenced Landsat imagery near both radar dates: optical support, a comparison, an unsupported water-loss interpretation, and an unresolved reference.",
        review_status="AI-assisted exploratory visual review; no independent human or field review",
        selection="Purposive examples selected after inspecting the saved overlay, optical imagery, quality masks, and exploratory water index. Not a random, blind, or held-out accuracy sample.",
        scope="A location-matched demonstration, not a detector accuracy benchmark or an estimate of confirmed changed area. No precision, recall, IoU, or full-scene accuracy is inferred.",
        limitations=[
            "Landsat dates are three days before Before radar and one day after After radar. Events within those gaps can differ.",
            "Landsat reference cells are 30 m; radar cells are 20 m. Both use EPSG:32640. Shared map bounds and nearest-neighbor display reprojection match locations, but physical subpixel registration has not been independently measured.",
            "A water index and Landsat water flags are diagnostic algorithms, not ground truth. Bright surfaces can produce misleading water-like index values.",
            "Visible flags count nonzero alpha in the saved overlay. Rounding hides some originally flagged cells, so this is a lower bound; absence of visible flags does not establish an unflagged model cell.",
            "A water-loss interpretation failure is not proof of a false positive for the generic radar-change detector. No detector false-positive/false-negative labels are established by this pilot.",
        ],
        methods=dict(quality="Exclude QA_PIXEL bits 0–5 (fill, dilated cloud, cirrus, cloud, cloud shadow, snow), any QA_RADSAT flag, and zero reflectance DN. Grey marks excluded cells.",
            reflectance_scale=SCALE, reflectance_offset=OFFSET,
            mndwi="(green − SWIR1) / (green + SWIR1); only positive denominators are used",
            water_like_cutoff=INDEX_THRESHOLD,
            water_like_note="Exploratory diagnostic fixed at 0.15 for these examples, not a validated water classifier or tuned detector threshold",
            source_water_flag="QA_PIXEL bit 7; independent source classification, not ground truth",
            display="Both dates share reflectance range 0–0.6 and gamma exponent 0.7. Natural colour uses red/green/blue; false colour uses SWIR1/NIR/red. Nearest-neighbor display on the 20 m radar grid adds no optical detail.",
            statistics="Optical diagnostics use original 30 m cell centres within each square. Radar medians are paired After − Before differences across all jointly valid native 20 m cells in the square, not just visible flags."),
        method_sources=[
            "https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands",
            "https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products",
            "https://planetarycomputer.microsoft.com/docs/concepts/sas/",
        ], sources=sources, radar_sources=sample.metadata["sources"], sites=sites,
    )
    files = {}
    csv_path = output / "sites.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["site_id", "outcome", "is_ground_truth", "west_m", "south_m", "east_m", "north_m",
                         "longitude", "latitude", "hh_paired_delta_db", "hv_paired_delta_db", "visible_flag_fraction_lower_bound",
                         "optical_before_date", "optical_after_date", "before_usable_fraction", "after_usable_fraction"])
        for site in sites:
            writer.writerow([site["id"], site["outcome"], "false", *site["bounds_wsen"], *site["center_lon_lat"],
                             *site["radar"]["median_delta_hh_hv_db"], site["radar"]["visible_flag_fraction"],
                             "2026-06-18", "2026-09-14", site["optical"]["before"]["usable_fraction"], site["optical"]["after"]["usable_fraction"]])
    files[str(csv_path.relative_to(ROOT))] = digest(csv_path)
    for directory in (source_out, images_out):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                files[str(path.relative_to(ROOT))] = digest(path)
    result["artifact_sha256"] = files
    (output / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")
    print("Built", output / "manifest.json")


if __name__ == "__main__":
    main()
