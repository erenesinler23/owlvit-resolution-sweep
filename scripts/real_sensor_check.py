"""Real-sensor check: native 1.2 m multispectral chips vs the simulated 1.2 m level.

SpaceNet 2 ships each chip as an 8-band WorldView-3 image at the sensor's native 1.2 m
(162 px) next to the 0.3 m pan-sharpened RGB. This runs the frozen detectors on the native
image (bands 5, 3, 2 = red, green, blue; upsampled to 650 px with bicubic) and compares F1
with the simulated 1.2 m level on the same chips and labels. Run after the test cache and
the frozen settings exist.

Brightness scale per city: 99.5th percentile of the pooled RGB values on 150 random chips
of that city (intensities only), rounded to 50. Same recipe as the PS-RGB constants.

    .venv/bin/python scripts/real_sensor_check.py --out results/test_real_sensor
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd
import rasterio
import yaml
from affine import Affine

from resolution_sweep import archive
from resolution_sweep.cache import colour_variants, load_config
from resolution_sweep.data import spacenet2
from resolution_sweep.detectors.colour import ColourRuleDetector
from resolution_sweep.detectors.owlvit import OwlRunner
from resolution_sweep.evaluate import evaluate_records, select_boxes
from resolution_sweep.matching import match_boxes, prf
from resolution_sweep.report import resolve

LEVEL = 1.2
BANDS = [5, 3, 2]          # red, green, blue in coastal, blue, green, yellow, red, ...


def mul_path(rec: dict) -> Path:
    return Path(rec["image_path"].replace("RGB-PanSharpen", "MUL"))


def city_scales(index: list, cities: list, n: int = 150, seed: int = 3) -> dict:
    rng = random.Random(seed)
    out = {}
    for city in cities:
        pool = [r for r in index if r["city"] == city]
        vals = []
        for rec in rng.sample(pool, n):
            with rasterio.open(mul_path(rec)) as ds:
                vals.append(ds.read(BANDS).reshape(3, -1)[:, ::3].ravel())
        v = np.concatenate(vals)
        out[city] = float(round(np.percentile(v, 99.5) / 50.0) * 50)
    return out


def read_native(rec: dict, scale: float) -> np.ndarray:
    with rasterio.open(mul_path(rec)) as ds:
        arr = ds.read(BANDS)
    u8 = spacenet2.to_uint8(arr, scale)
    rgb = np.dstack([u8[0], u8[1], u8[2]])
    return cv2.resize(rgb, (650, 650), interpolation=cv2.INTER_CUBIC)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, blocks: np.ndarray, n_boot: int, seed: int):
    """a, b: (chips, 3) counts. Returns F1 of each and bootstrap CIs, in percent."""
    uniq, inv = np.unique(blocks, return_inverse=True)
    sa = np.zeros((len(uniq), 3))
    sb = np.zeros((len(uniq), 3))
    for i, k in enumerate(inv):
        sa[k] += a[i]
        sb[k] += b[i]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    fa = prf(*sa[idx].sum(axis=1).T)[2]
    fb = prf(*sb[idx].sum(axis=1).T)[2]
    d = fa - fb

    def ci(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    point = (float(prf(*sa.sum(axis=0))[2]), float(prf(*sb.sum(axis=0))[2]))
    return point, ci(fa), ci(fb), ci(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--frozen", default="configs/frozen.yaml")
    ap.add_argument("--cache", default="archive/test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    frozen = yaml.safe_load(open(args.frozen))["detectors"]
    gsd = cfg["dataset"]["native_gsd_m"]
    index = list(csv.DictReader(open("archive/index.csv")))
    have = {p.stem for p in Path(args.cache).rglob("*.npz")}
    records = [r for r in index if r["chip_id"] in have and r["split"] == args.split]
    if args.limit:
        records = records[: args.limit]
    scales = city_scales(index, cfg["dataset"]["cities"])
    print("MUL scales", scales, "chips", len(records), flush=True)

    prompts = tuple(cfg["detectors"]["prompts"])
    runners = {m: OwlRunner(cfg["detectors"][m]["model_id"], cfg["detectors"][m].get("revision"),
                            prompts) for m in ("owlvit", "owlv2")}
    variants = colour_variants(cfg)
    cvar = frozen["colour"]["fixed"].split("|")[0]
    metric_tf = Affine.scale(gsd, gsd)
    native = {}
    for i, rec in enumerate(records, 1):
        img = read_native(rec, scales[rec["city"]])
        arrays = {"extent_m": np.array([650 * gsd, 650 * gsd], dtype=np.float32)}
        for m, runner in runners.items():
            raw = runner.raw(img)
            arrays[archive.array_key(m, "mul", LEVEL, "boxes")] = (raw.boxes_px * gsd).astype(np.float32)
            arrays[archive.array_key(m, "mul", LEVEL, "scores")] = raw.scores
        found = ColourRuleDetector(**variants[cvar]).detect(
            img[..., 0], img[..., 1], img[..., 2], metric_tf, None)
        arrays[archive.array_key(f"colour-{cvar}", "mul", LEVEL, "boxes")] = np.array(
            [d.geometry.bounds for d in found], dtype=np.float32).reshape(-1, 4)
        native[rec["chip_id"]] = arrays
        if i % 50 == 0:
            print(f"{i}/{len(records)}", flush=True)

    refs = {r["chip_id"]: archive.load_chip(Path(args.cache), r["city"], r["chip_id"])[1]
            for r in records}
    blocks = np.array([r["block"] for r in records])
    order = [r["chip_id"] for r in records]
    rows = []
    for det in ("owlvit", "owlv2", "colour"):
        key, ec = resolve(det, frozen[det]["fixed"], cfg)
        for iou in cfg["evaluation"]["iou_thresholds"]:
            nat = []
            for cid in order:
                boxes, _ = select_boxes(native[cid], key, "mul", LEVEL, ec)
                res = match_boxes(boxes, refs[cid], iou)
                nat.append([res.matches, res.predictions, res.references])
            nat = np.array(nat, dtype=float)
            for method in cfg["sweep"]["methods"]:
                df = evaluate_records(records, Path(args.cache), key, method, [LEVEL],
                                      {"fixed": ec}, [iou])
                sim = df.set_index("chip_id").loc[order, ["matches", "predictions", "references"]]
                (f_nat, f_sim), ci_nat, ci_sim, ci_diff = paired_bootstrap(
                    nat, sim.to_numpy(dtype=float), blocks,
                    cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
                rows.append({"detector": det, "iou": iou, "sim_method": method,
                             "native_f1": f_nat, "native_ci": ci_nat,
                             "sim_f1": f_sim, "sim_ci": ci_sim,
                             "diff_native_minus_sim": f_nat - f_sim, "diff_ci": ci_diff})
                print(det, iou, method, round(f_nat, 1), round(f_sim, 1), ci_diff, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "real_sensor.csv", index=False)
    (out / "scales.json").write_text(json.dumps(scales))


if __name__ == "__main__":
    main()
