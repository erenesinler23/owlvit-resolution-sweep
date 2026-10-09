"""Add colour-rule variants to an existing cache without touching the OWL arrays.

The test cache was built before the colour variant was frozen, so it holds only the paper's
fixed thresholds. This adds the frozen variant by merging new arrays into each chip file.
Only image data is read here, never labels.

    .venv/bin/python scripts/add_colour_variant.py --cache archive/test --variants r120_c15
"""
import argparse
from pathlib import Path

import numpy as np
from affine import Affine

from resolution_sweep import archive
from resolution_sweep.cache import colour_variants, load_config, scale_for
from resolution_sweep.data import spacenet2
from resolution_sweep.degrade import degrade
from resolution_sweep.detectors.colour import ColourRuleDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds, sw = cfg["dataset"], cfg["sweep"]
    gsd = ds["native_gsd_m"]
    variants = colour_variants(cfg)
    index = {}
    import csv
    for r in csv.DictReader(open("archive/index.csv")):
        index[r["chip_id"]] = r
    metric_tf = Affine.scale(gsd, gsd)
    files = sorted(Path(args.cache).rglob("*.npz"))
    for i, npz in enumerate(files, 1):
        rec = index[npz.stem]
        arrays, refs = archive.load_chip(Path(args.cache), rec["city"], rec["chip_id"])
        r, g, b, _, _ = spacenet2.read_rgb(Path(rec["image_path"]), scale_for(ds, rec["city"]))
        rgb = np.dstack([r, g, b])
        for method in sw["methods"]:
            for level in sw["levels_gsd_m"]:
                img = degrade(rgb, gsd, level, method, sw["upsample"], sw["blur_sigma_factor"])
                for name in args.variants:
                    det = ColourRuleDetector(**variants[name])
                    found = det.detect(img[..., 0], img[..., 1], img[..., 2], metric_tf, None)
                    boxes = np.array([d.geometry.bounds for d in found],
                                     dtype=np.float32).reshape(-1, 4)
                    arrays[archive.array_key(f"colour-{name}", method, level, "boxes")] = boxes
        archive.save_chip(Path(args.cache), rec["city"], rec["chip_id"], arrays, refs)
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
