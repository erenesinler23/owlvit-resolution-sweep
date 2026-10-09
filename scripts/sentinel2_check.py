"""Real Sentinel-2 vs simulated 10 m on identical chips and labels.

Reads the crops written by fetch_sentinel2.py (20 by 20 px over each chip footprint), converts
them to 8 bit with one constant per city (99.5th percentile of the pooled RGB digital numbers,
rounded to 50, same recipe as the aerial scales), upsamples to 650 px with bicubic like the
simulated levels, and runs the frozen detectors. Scores against the SpaceNet building labels and
compares with the simulated 10 m level (area and blur_resize) on the same chips, with a paired
block bootstrap. Also reports the oracle-location baseline on the real crops.

    .venv/bin/python scripts/sentinel2_check.py --out results/test_sentinel2 --cities AOI_3_Paris AOI_5_Khartoum
"""
import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import yaml
from affine import Affine

from real_sensor_check import paired_bootstrap
from resolution_sweep import archive
from resolution_sweep.baseline import chance_matches
from resolution_sweep.cache import colour_variants, load_config
from resolution_sweep.data import spacenet2
from resolution_sweep.detectors.colour import ColourRuleDetector
from resolution_sweep.detectors.owlvit import OwlRunner
from resolution_sweep.evaluate import evaluate_records, select_boxes
from resolution_sweep.matching import match_boxes
from resolution_sweep.report import resolve

LEVEL = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--frozen", default="configs/frozen.yaml")
    ap.add_argument("--cache", default="archive/test")
    ap.add_argument("--s2", default="data/sentinel2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    frozen = yaml.safe_load(open(args.frozen))["detectors"]
    gsd = cfg["dataset"]["native_gsd_m"]
    index = list(csv.DictReader(open("archive/index.csv")))
    have = {p.stem for p in Path(args.cache).rglob("*.npz")}
    records = [r for r in index if r["chip_id"] in have and r["split"] == args.split
               and r["city"] in args.cities and (Path(args.s2) / f"{r['chip_id']}.npy").exists()]
    if args.limit:
        records = records[: args.limit]
    print(len(records), "chips with a Sentinel-2 crop", flush=True)

    crops = {r["chip_id"]: np.load(Path(args.s2) / f"{r['chip_id']}.npy") for r in records}
    scales = {}
    for city in args.cities:
        v = np.concatenate([crops[r["chip_id"]].reshape(-1) for r in records if r["city"] == city])
        scales[city] = float(round(np.percentile(v, 99.5) / 50.0) * 50)
    print("scales", scales, flush=True)

    prompts = tuple(cfg["detectors"]["prompts"])
    runners = {m: OwlRunner(cfg["detectors"][m]["model_id"], cfg["detectors"][m].get("revision"),
                            prompts) for m in ("owlvit", "owlv2")}
    variants = colour_variants(cfg)
    cvars = {frozen["colour"]["fixed"].split("|")[0]}
    for v in frozen["colour"]["per_level"].values():
        cvars.add(v.split("|")[0])
    metric_tf = Affine.scale(gsd, gsd)
    native = {}
    for i, rec in enumerate(records, 1):
        u8 = spacenet2.to_uint8(np.moveaxis(crops[rec["chip_id"]], -1, 0), scales[rec["city"]])
        small = np.dstack([u8[0], u8[1], u8[2]])
        img = cv2.resize(small, (650, 650), interpolation=cv2.INTER_CUBIC)
        arrays = {"extent_m": np.array([650 * gsd, 650 * gsd], dtype=np.float32)}
        for m, runner in runners.items():
            raw = runner.raw(img)
            arrays[archive.array_key(m, "s2", LEVEL, "boxes")] = (raw.boxes_px * gsd).astype(np.float32)
            arrays[archive.array_key(m, "s2", LEVEL, "scores")] = raw.scores
        for cvar in cvars:
            found = ColourRuleDetector(**variants[cvar]).detect(
                img[..., 0], img[..., 1], img[..., 2], metric_tf, None)
            arrays[archive.array_key(f"colour-{cvar}", "s2", LEVEL, "boxes")] = np.array(
                [d.geometry.bounds for d in found], dtype=np.float32).reshape(-1, 4)
        native[rec["chip_id"]] = arrays
        if i % 50 == 0:
            print(f"{i}/{len(records)}", flush=True)

    refs = {r["chip_id"]: archive.load_chip(Path(args.cache), r["city"], r["chip_id"])[1]
            for r in records}
    order = [r["chip_id"] for r in records]
    blocks = np.array([r["block"] for r in records])
    rows = []
    for det in ("owlvit", "owlv2", "colour"):
        settings = {"fixed": frozen[det]["fixed"]}
        pl = frozen[det]["per_level"]
        settings["per_level"] = pl[str(LEVEL)] if str(LEVEL) in pl else pl[LEVEL]
        for sname, name in settings.items():
            key, ec = resolve(det, name, cfg)
            for iou in cfg["evaluation"]["iou_thresholds"]:
                nat, orc = [], []
                for cid in order:
                    boxes, _ = select_boxes(native[cid], key, "s2", LEVEL, ec)
                    res = match_boxes(boxes, refs[cid], iou)
                    nat.append([res.matches, res.predictions, res.references])
                    seed = zlib.crc32(f"{cid}|{det}|{iou}".encode())
                    orc.append([chance_matches(boxes, refs[cid], native[cid]["extent_m"], iou, 10,
                                               seed, "oracle"), res.predictions, res.references])
                nat = np.array(nat, dtype=float)
                orc = np.array(orc, dtype=float)
                for method in cfg["sweep"]["methods"]:
                    df = evaluate_records(records_dicts(records), Path(args.cache), key, method,
                                          [LEVEL], {"cfg": ec}, [iou])
                    sim = df.set_index("chip_id").loc[order, ["matches", "predictions", "references"]]
                    (f_nat, f_sim), ci_nat, ci_sim, ci_diff = paired_bootstrap(
                        nat, sim.to_numpy(dtype=float), blocks,
                        cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
                    (_, f_orc), _, ci_orc, _ = paired_bootstrap(
                        nat, orc, blocks, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
                    rows.append({"detector": det, "setting": sname, "iou": iou, "sim_method": method,
                                 "s2_f1": f_nat, "s2_ci": ci_nat, "sim_f1": f_sim, "sim_ci": ci_sim,
                                 "diff_s2_minus_sim": f_nat - f_sim, "diff_ci": ci_diff,
                                 "oracle_f1_on_s2": f_orc, "n_chips": len(order),
                                 "s2_predictions": int(nat[:, 1].sum()), "s2_matches": int(nat[:, 0].sum())})
                    print(det, sname, iou, method, round(f_nat, 1), round(f_sim, 1), round(f_orc, 1),
                          ci_diff, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sentinel2.csv", index=False)
    (out / "scales.json").write_text(json.dumps(scales))


def records_dicts(records):
    return records


if __name__ == "__main__":
    main()
