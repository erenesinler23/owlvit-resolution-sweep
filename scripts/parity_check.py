"""Parity check against a saved archive from the original paper (read-only).

Replays one outputs/verified/<city> folder through the NEW code:
  1. Matcher parity: the archive's own saved detections vs its references.
  2. Colour rule parity: rebuild detections from image.tif and compare counts.

    .venv/bin/python scripts/parity_check.py ~/vscode-workspace/geoai/outputs/verified/ankara

Nothing is written to the archive. Needs the "parity" extras (geopandas, rasterio).
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from resolution_sweep.detectors.colour import ColourRuleDetector
from resolution_sweep.matching import match, size_mask


def clean(geoms):
    out = []
    for g in geoms:
        g = g.buffer(0) if not g.is_valid else g
        if not g.is_empty and g.area > 0:
            out.append(g)
    return out


def bounds_arr(geoms):
    return np.array([g.bounds for g in geoms], dtype=float).reshape(-1, 4)


def evaluate(pred_geoms, ref_geoms, max_size_m, iou):
    keep = size_mask(bounds_arr(pred_geoms), max_size_m)
    kept = [g for g, k in zip(pred_geoms, keep) if k]
    return match(kept, ref_geoms, iou), int((~keep).sum())


def main(path: str) -> int:
    root = Path(path).expanduser()
    expected = {r["detector_name"]: r for r in json.loads((root / "results.json").read_text())}
    settings = json.loads((root / "manifest.json").read_text())["settings"]
    max_size, iou = settings["max_size_m"], settings["iou_threshold"]
    exp = expected["naive"]
    print(f"Archive: {root.name}  expected naive: raw={exp['n_raw']} kept={exp['n_total']} "
          f"matched={exp['n_matched']} refs={exp['n_buildings_total']}")

    with rasterio.open(root / "image.tif") as src:
        bands = src.read([1, 2, 3])
        tf, crs = src.transform, src.crs
    print(f"image.tif dtype={bands.dtype} crs={crs.to_string()} shape={bands.shape}")

    refs = gpd.read_file(root / "buildings.geojson").to_crs(crs)
    refs_geoms = list(refs.geometry)
    valid = gpd.read_file(root / "valid_area.geojson").to_crs(crs).geometry.union_all()

    ok = True

    # 1. Matcher parity on the archive's saved detections.
    saved = gpd.read_file(root / "naive_detections.geojson").to_crs(crs)
    res, removed = evaluate(clean(list(saved.geometry)), refs_geoms, max_size, iou)
    print(f"[matcher] saved detections: kept={res.predictions} removed={removed} "
          f"matched={res.matches} refs={res.references}")
    ok &= res.matches == exp["n_matched"] and res.references == exp["n_buildings_total"]

    # 2. Colour rule parity from the image.
    mask = rasterize([valid], out_shape=bands.shape[1:], transform=tf, fill=0, default_value=1).astype(bool)
    r, g, b = (np.where(mask, band, 0) for band in bands)
    dets = ColourRuleDetector(settings["red_min"], settings["red_minus_green_min"],
                              settings["red_minus_blue_min"]).detect(r, g, b, tf, crs)
    clipped = clean([d.geometry.intersection(valid) for d in dets])
    res2, removed2 = evaluate(clipped, refs_geoms, max_size, iou)
    print(f"[colour ] rebuilt: raw={len(clipped)} removed={removed2} kept={res2.predictions} "
          f"matched={res2.matches}")
    ok &= (len(clipped) == exp["n_raw"] and res2.predictions == exp["n_total"]
           and res2.matches == exp["n_matched"])

    print("PARITY OK" if ok else "PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
