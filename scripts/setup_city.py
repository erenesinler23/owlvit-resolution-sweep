"""Prepare an extra held-out city: brightness scale, config and index rows.

All chips of the city go to the test split. The city is never used for tuning. The brightness
constant uses the same recipe as Paris and Khartoum (99.5th percentile of pooled RGB values,
intensities only, rounded to 50). Appends rows to archive/index.csv and writes configs/<name>.yaml.

    .venv/bin/python scripts/setup_city.py --city AOI_2_Vegas --name vegas
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import yaml

from resolution_sweep.data import spacenet2
from resolution_sweep.split import block_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/sweep.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]
    chips = spacenet2.index_city(Path(ds["root"]), args.city, ds["image_glob"], ds["label_glob"])
    print(len(chips), "chips with image and label")
    vals = []
    for c in random.Random(3).sample(chips, min(150, len(chips))):
        with rasterio.open(c.image_path) as src:
            vals.append(src.read([1, 2, 3]).reshape(3, -1)[:, ::3].ravel())
    scale = float(round(np.percentile(np.concatenate(vals), 99.5) / 50.0) * 50)
    print("scale", scale)

    rows = []
    for c in chips:
        with rasterio.open(c.image_path) as src:
            x, y = spacenet2.chip_centre(src.transform, src.width, src.height)
        rows.append({"chip_id": c.chip_id, "city": c.city,
                     "block": block_key(c.city, x, y, cfg["split"]["block_cell"]), "split": "test",
                     "image_path": str(c.image_path), "label_path": str(c.label_path)})
    path = Path("archive/index.csv")
    existing = list(csv.DictReader(open(path)))
    have = {r["chip_id"] for r in existing}
    new = [r for r in rows if r["chip_id"] not in have]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(existing[0].keys()))
        w.writeheader()
        w.writerows(existing + new)
    print("appended", len(new), "rows to", path)

    cfg["dataset"]["cities"] = [args.city]
    cfg["dataset"]["scale_max"] = {args.city: scale}
    cfg["split"]["n_test"] = len(chips)
    out = Path(f"configs/{args.name}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
