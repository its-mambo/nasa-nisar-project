"""Optional geospatial audit; run with evaluation/requirements.txt installed."""
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import rasterio

from build_matched_evidence import optical_metrics, ROOT


class OpticalPilotTests(unittest.TestCase):
    def test_quality_mask_excludes_each_bad_bit_and_keeps_water_flag(self):
        raw = np.full((7, 1, 9), 15000, dtype=np.uint16)
        raw[5] = [[0, 1, 2, 4, 8, 16, 32, 128, 0]]
        raw[6] = 0
        raw[6, 0, -1] = 1
        result = optical_metrics(raw)
        self.assertEqual(result["usable_cells"], 2)
        self.assertEqual(result["source_water_flag_cells"], 1)
        raw[0, 0, 0] = 0
        self.assertEqual(optical_metrics(raw)["usable_cells"], 1)

    def test_native_geotiffs_reproduce_every_diagnostic_without_upsampling(self):
        record = json.loads((ROOT / "evaluation/karakalpakstan-pilot/manifest.json").read_text(encoding="utf-8"))
        for site in record["sites"]:
            transforms = []
            for label in ("before", "after"):
                path = ROOT / f"evaluation/karakalpakstan-pilot/source/{site['id']}-{label}.tif"
                with self.subTest(site=site["id"], date=label), rasterio.open(path) as src:
                    self.assertEqual(src.crs.to_epsg(), 32640)
                    self.assertEqual(src.res, (30, 30))
                    self.assertEqual(src.shape, (100, 100))
                    self.assertEqual(src.descriptions, ("red", "green", "blue", "nir08", "swir16", "qa_pixel", "qa_radsat"))
                    transforms.append(src.transform)
                    actual = optical_metrics(src.read())
                    expected = {k: v for k, v in site["optical"][label].items() if k != "transform"}
                    self.assertEqual(actual, expected)
            self.assertEqual(transforms[0], transforms[1])
            for url in site["images"].values():
                with Image.open(ROOT / "frontend/features/results/reference/matched" / url.split("/")[-1]) as image:
                    self.assertEqual(image.size, (150, 150))


if __name__ == "__main__":
    unittest.main()
