"""Cache phase: degrade each chip, run the detectors, save raw candidates.

Usage (from the repo root, inside the virtual environment):
    python -m resolution_sweep.cache --config configs/sweep.yaml --split dev --limit 20

Nothing here applies thresholds. Evaluation happens later from the archive.
Untested against real data: the first run on 20 chips is the integration test.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import yaml
from affine import Affine

from . import archive
from .data import spacenet2
from .degrade import degrade
from .detectors.colour import ColourRuleDetector
from .split import assign_split, subsample


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def colour_variants(cfg: dict) -> dict:
    c = cfg["detectors"]["colour"]
    variants = {"paper_fixed": dict(c["paper_fixed"])}
    grid = c["dev_grid"]
    for red, chroma in itertools.product(grid["red_min"], grid["chroma_min"]):
        variants[f"r{red}_c{chroma}"] = {"red_min": red, "rg_min": chroma, "rb_min": chroma}
    return variants


def build_index(cfg: dict, index_path: Path) -> list[dict]:
    if index_path.exists():
        with open(index_path) as fh:
            return list(csv.DictReader(fh))
    import rasterio
    ds = cfg["dataset"]
    root = Path(ds["root"])
    chips = []
    for city in ds["cities"]:
        chips += spacenet2.index_city(root, city, ds["image_glob"], ds["label_glob"])
    if not chips:
        raise SystemExit("No chips found. Check dataset.root and the globs in the config.")
    xs, ys = [], []
    for c in chips:
        with rasterio.open(c.image_path) as src:
            x, y = spacenet2.chip_centre(src.transform, src.width, src.height)
        xs.append(x)
        ys.append(y)
    sp = cfg["split"]
    recs = assign_split([c.chip_id for c in chips], [c.city for c in chips], xs, ys,
                        sp["block_cell"], sp["dev_fraction"], sp["seed"])
    by_id = {c.chip_id: c for c in chips}
    for r in recs:
        r["image_path"] = str(by_id[r["chip_id"]].image_path)
        r["label_path"] = str(by_id[r["chip_id"]].label_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        writer.writeheader()
        writer.writerows(recs)
    return recs


def scale_for(ds: dict, city: str):
    """scale_max may be one number or a per-city mapping."""
    sm = ds.get("scale_max")
    return sm.get(city) if isinstance(sm, dict) else sm


def process_chip(rec: dict, cfg: dict, out_dir: Path, colour_names: list, runners: dict) -> None:
    ds, sw = cfg["dataset"], cfg["sweep"]
    gsd = ds["native_gsd_m"]
    r, g, b, tf, _ = spacenet2.read_rgb(Path(rec["image_path"]), scale_for(ds, rec["city"]))
    h, w = r.shape
    rgb = np.dstack([r, g, b])
    refs = spacenet2.read_refs(Path(rec["label_path"]), tf, gsd, w, h)
    metric_tf = Affine.scale(gsd, gsd)
    variants = colour_variants(cfg)
    arrays: dict = {"extent_m": np.array([w * gsd, h * gsd], dtype=np.float32)}
    for method in sw["methods"]:
        for level in sw["levels_gsd_m"]:
            img = degrade(rgb, gsd, level, method, sw["upsample"], sw["blur_sigma_factor"])
            for name in colour_names:
                det = ColourRuleDetector(**variants[name])
                found = det.detect(img[..., 0], img[..., 1], img[..., 2], metric_tf, None)
                boxes = np.array([d.geometry.bounds for d in found], dtype=np.float32).reshape(-1, 4)
                arrays[archive.array_key(f"colour-{name}", method, level, "boxes")] = boxes
            for mname, runner in runners.items():
                raw = runner.raw(img)
                arrays[archive.array_key(mname, method, level, "boxes")] = (
                    raw.boxes_px * gsd).astype(np.float32)
                arrays[archive.array_key(mname, method, level, "scores")] = raw.scores
    archive.save_chip(out_dir, rec["city"], rec["chip_id"], arrays, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--models", nargs="*", default=["owlvit"], choices=["owlvit", "owlv2"])
    ap.add_argument("--colour-variants", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cities", nargs="*", default=None,
                    help="restrict to these cities (default: all in the index)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out or f"archive/{args.split}")
    index = build_index(cfg, Path("archive/index.csv"))
    pool = [r for r in index if r["split"] == args.split
            and (not args.cities or r["city"] in args.cities)]
    n = cfg["split"]["n_dev" if args.split == "dev" else "n_test"]
    chips = subsample(pool, n, cfg["split"]["seed"])
    if args.limit:
        chips = chips[: args.limit]

    names = args.colour_variants or (
        list(colour_variants(cfg)) if args.split == "dev" else ["paper_fixed"])
    runners = {}
    if args.models:
        from .detectors.owlvit import OwlRunner
        prompts = tuple(cfg["detectors"]["prompts"])
        for m in args.models:
            spec = cfg["detectors"][m]
            runners[m] = OwlRunner(spec["model_id"], spec.get("revision"), prompts)

    done = 0
    for rec in chips:
        npz, _ = archive.chip_paths(out_dir, rec["city"], rec["chip_id"])
        if npz.exists() and not args.force:
            continue
        t0 = time.time()
        process_chip(rec, cfg, out_dir, names, runners)
        done += 1
        print(f"[{done}/{len(chips)}] {rec['chip_id']} {time.time() - t0:.1f}s", flush=True)

    archive.write_manifest(out_dir, cfg, {
        "split": args.split, "models": args.models, "colour_variants": names,
        "n_chips_requested": len(chips)})


if __name__ == "__main__":
    main()
