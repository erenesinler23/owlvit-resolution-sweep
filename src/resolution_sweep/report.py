"""Final report on a cache split using the frozen settings.

    python -m resolution_sweep.report --cache archive/test --split test --out results/test

Writes per-detector curve tables (point, bootstrap CI, chance F1), break points with
CIs, a per-level-setting upper bound, and the F1-versus-resolution figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analysis import above_chance, break_point, plot_f1_curves
from .bootstrap import bootstrap_f1_curves
from .cache import load_config
from .evaluate import (EvalConfig, chance_f1_by_level, counts_tensor, evaluate_records,
                       owl_grid)


def resolve(det: str, name: str, cfg: dict) -> tuple[str, EvalConfig]:
    """Return (cache detector key, EvalConfig) for a frozen config name."""
    if det == "colour":
        variant, size = name.split("|")
        return f"colour-{variant}", EvalConfig((), 0.0, float(size.replace("max", "")), 0.0)
    return det, owl_grid(cfg)[name]


def curve_rows(df, config, method, iou, n_boot, seed, alpha):
    levels, counts, blocks = counts_tensor(df, config, method, iou)
    point, boot = bootstrap_f1_curves(counts, blocks, n_boot, seed)
    clv, chance = chance_f1_by_level(df, config, method, iou)
    assert list(clv) == list(levels)
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    bp = break_point(levels, point, 0.5, chance, ref="peak")
    bps = np.array([np.nan if (b := break_point(levels, row, 0.5, chance, ref="peak")) is None else b
                    for row in boot])
    ok = np.isfinite(bps)
    bp_f = break_point(levels, point, 0.5, chance, ref="finest")
    bps_f = np.array([np.nan if (b := break_point(levels, row, 0.5, chance, ref="finest")) is None else b
                      for row in boot])
    ok_f = np.isfinite(bps_f)
    out = {
        "levels": [float(x) for x in levels], "f1": point, "lo": lo, "hi": hi, "chance": chance,
        "break_point": bp, "break_ci": (
            [float(np.nanpercentile(bps, 100 * alpha / 2)), float(np.nanpercentile(bps, 100 * (1 - alpha / 2)))]
            if ok.any() else None),
        "break_found_frac": float(ok.mean()),
        "break_point_finest_ref": bp_f,
        "break_ci_finest_ref": (
            [float(np.nanpercentile(bps_f, 100 * alpha / 2)), float(np.nanpercentile(bps_f, 100 * (1 - alpha / 2)))]
            if ok_f.any() else None),
        "above_chance_finest": above_chance(boot[:, 0], float(chance[0]), alpha),
        "above_chance_peak": above_chance(boot[:, int(np.nanargmax(point - chance))],
                                          float(chance[int(np.nanargmax(point - chance))]), alpha),
        "peak_level": float(levels[int(np.nanargmax(point))]),
        "peak_f1": float(np.nanmax(point)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--frozen", default="configs/frozen.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chance-reps", type=int, default=20)
    ap.add_argument("--detectors", nargs="+", default=["owlvit", "owlv2", "colour"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    frozen = yaml.safe_load(open(args.frozen))["detectors"]
    bs = cfg["bootstrap"]
    levels = cfg["sweep"]["levels_gsd_m"]
    index = pd.read_csv("archive/index.csv").to_dict("records")
    have = {p.stem for p in Path(args.cache).rglob("*.npz")}
    records = [r for r in index if r["chip_id"] in have and r["split"] == args.split]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} {args.split} chips")

    summary, curves = {}, {}
    for det in args.detectors:
        fz = frozen[det]
        for method in cfg["sweep"]["methods"]:
            for iou in cfg["evaluation"]["iou_thresholds"]:
                # fixed setting
                key, ec = resolve(det, fz["fixed"], cfg)
                df = evaluate_records(records, Path(args.cache), key, method, levels,
                                      {"fixed": ec}, [iou], chance_reps=args.chance_reps)
                df.assign(label=det).to_csv(out / f"counts_{det}_{method}_iou{iou}_fixed.csv", index=False)
                r = curve_rows(df, "fixed", method, iou, bs["n_boot"], bs["seed"], bs["alpha"])
                summary[f"{det}|{method}|iou{iou}|fixed"] = {
                    k: v for k, v in r.items() if k not in ("f1", "lo", "hi", "chance")}
                summary[f"{det}|{method}|iou{iou}|fixed"]["setting"] = fz["fixed"]
                pd.DataFrame({k: r[k] for k in ("levels", "f1", "lo", "hi", "chance")}).to_csv(
                    out / f"curve_{det}_{method}_iou{iou}_fixed.csv", index=False)
                curves[(det, method, iou, "fixed")] = r
                # per-level setting (upper bound)
                parts = []
                for lv in levels:
                    key, ec = resolve(det, fz["per_level"][float(lv)], cfg)
                    d = evaluate_records(records, Path(args.cache), key, method, [lv],
                                         {"perlevel": ec}, [iou], chance_reps=args.chance_reps)
                    parts.append(d)
                dfp = pd.concat(parts)
                rp = curve_rows(dfp, "perlevel", method, iou, bs["n_boot"], bs["seed"], bs["alpha"])
                summary[f"{det}|{method}|iou{iou}|per_level"] = {
                    k: v for k, v in rp.items() if k not in ("f1", "lo", "hi", "chance")}
                pd.DataFrame({k: rp[k] for k in ("levels", "f1", "lo", "hi", "chance")}).to_csv(
                    out / f"curve_{det}_{method}_iou{iou}_per_level.csv", index=False)
                curves[(det, method, iou, "per_level")] = rp
                print(det, method, iou, "break", r["break_point"], "peak", r["peak_level"],
                      round(r["peak_f1"], 1), flush=True)
                (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for method in cfg["sweep"]["methods"]:
        for iou in cfg["evaluation"]["iou_thresholds"]:
            fig = {d: {"gsd": curves[(d, method, iou, "fixed")]["levels"],
                       "f1": curves[(d, method, iou, "fixed")]["f1"],
                       "lo": curves[(d, method, iou, "fixed")]["lo"],
                       "hi": curves[(d, method, iou, "fixed")]["hi"]}
                   for d in args.detectors}
            plot_f1_curves(fig, str(out / f"f1_vs_resolution_{method}_iou{iou}.png"),
                           f"{args.split}: F1 vs ground sampling distance ({method}, IoU {iou})")
    print("done ->", out)


if __name__ == "__main__":
    main()
