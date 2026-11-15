"""Fetch public, georeferenced Landsat reference subsets; never persist SAS tokens."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ITEMS = {
    "before": "LC08_L2SP_162030_20260618_02_T1",
    "after": "LC09_L2SP_162030_20260914_02_T1",
}
BANDS = ("red", "green", "blue", "nir08", "swir16", "qa_pixel", "qa_radsat")
BOUNDS = (539900, 4761720, 658540, 4834760)


def fetch_json(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = {}
    for label, ident in ITEMS.items():
        item = fetch_json(f"{STAC}/collections/landsat-c2-l2/items/{ident}")
        (args.out / f"{label}-item.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        items[label] = item
    token = fetch_json("https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2")["token"]

    def download(task):
        label, band = task
        output = args.out / f"{label}-{band}.tif"
        if output.exists():
            return
        href = items[label]["assets"][band]["href"]
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=45,
                          GDAL_HTTP_MAX_RETRY=2, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF"):
            with rasterio.open(href + "?" + token) as src:
                if src.crs.to_epsg() != 32640:
                    raise ValueError("Expected the reference source in EPSG:32640")
                raw = from_bounds(*BOUNDS, transform=src.transform)
                window = Window(np.floor(raw.col_off), np.floor(raw.row_off),
                                np.ceil(raw.width) + 2, np.ceil(raw.height) + 2)
                window = window.intersection(Window(0, 0, src.width, src.height))
                data = src.read(1, window=window)
                profile = src.profile.copy()
                profile.update(width=data.shape[1], height=data.shape[0],
                               transform=src.window_transform(window), driver="GTiff", compress="deflate")
                with rasterio.open(output, "w", **profile) as dst:
                    dst.write(data, 1)
        print("Saved", output.name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [(label, band) for label in ITEMS for band in BANDS]))


if __name__ == "__main__":
    main()
