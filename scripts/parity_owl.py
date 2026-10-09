"""OWL-ViT parity check against a saved archive (read-only). UNTESTED: needs torch.

Runs the NEW OwlRunner on the archive's masked image and compares its boxes and
per-box best score with the archive's owlvit_raw.json. Differences beyond float
noise point to a bug in the wrapper.

    .venv/bin/python scripts/parity_owl.py ~/vscode-workspace/geoai/outputs/verified/ankara
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from resolution_sweep.detectors.owlvit import OwlRunner


def main(path: str) -> int:
    root = Path(path).expanduser()
    settings = json.loads((root / "manifest.json").read_text())["settings"]
    saved = json.loads((root / "owlvit_raw.json").read_text())
    with rasterio.open(root / "image.tif") as src:
        bands, tf, crs = src.read([1, 2, 3]), src.transform, src.crs
    valid = gpd.read_file(root / "valid_area.geojson").to_crs(crs).geometry.union_all()
    mask = rasterize([valid], out_shape=bands.shape[1:], transform=tf, fill=0, default_value=1).astype(bool)
    rgb = np.dstack([np.where(mask, band, 0) for band in bands]).astype(np.uint8)

    runner = OwlRunner("google/owlvit-base-patch32", settings["model_revision"],
                       tuple(settings["text_queries"]))
    raw = runner.raw(rgb)

    ref_boxes = np.array(saved["boxes"], dtype=float)
    ref_scores = np.array(saved["scores"], dtype=float)
    box_err = np.abs(raw.boxes_px - ref_boxes).max()
    score_err = np.abs(raw.scores.max(axis=1) - ref_scores).max()
    print(f"boxes: {raw.boxes_px.shape} vs {ref_boxes.shape}, max abs diff {box_err:.4f} px")
    print(f"scores: max abs diff {score_err:.5f}")
    ok = raw.boxes_px.shape == ref_boxes.shape and box_err < 0.5 and score_err < 1e-3
    print("OWL PARITY OK" if ok else "OWL PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
