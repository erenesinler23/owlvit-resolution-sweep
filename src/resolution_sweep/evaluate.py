"""Evaluation phase: apply prompt, threshold and size settings to the cache.

All settings are applied after the forward pass, so tuning on the development
split costs matching time only. Counts are kept per chip so the bootstrap can
resample blocks.

    python -m resolution_sweep.evaluate tune --cache archive/dev --detector owlvit
"""
from __future__ import annotations

import argparse
import itertools
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import archive
from .baseline import chance_matches
from .cache import load_config
from .matching import match_boxes, prf, size_mask


@dataclass(frozen=True)
class EvalConfig:
    prompt_idx: tuple = ()
    score_threshold: float = 0.0
    max_size_m: Optional[float] = None
    min_area_m2: float = 0.0


def select_boxes(arrays: dict, detector: str, method: str, level: float, cfg: EvalConfig):
    boxes = arrays[archive.array_key(detector, method, level, "boxes")]
    if detector.startswith("owl"):
        if not cfg.prompt_idx:
            raise ValueError("owl detectors need at least one prompt index")
        scores = arrays[archive.array_key(detector, method, level, "scores")]
        boxes = boxes[scores[:, list(cfg.prompt_idx)].max(axis=1) >= cfg.score_threshold]
    extent = arrays.get("extent_m")
    if extent is not None:
        # The original pipeline clips boxes to the image and drops empty ones
        # before the size filter. Same order here.
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, extent[0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, extent[1])
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    keep = size_mask(boxes, cfg.max_size_m, cfg.min_area_m2)
    return boxes[keep], int((~keep).sum())


def evaluate_records(records: list, cache_dir: Path, detector: str, method: str,
                     levels: list, configs: dict, iou_thresholds: list,
                     chance_reps: int = 0) -> pd.DataFrame:
    """Per-chip counts for every (config, level, IoU). Chips outer to load each file once."""
    rows = []
    for rec in records:
        arrays, refs = archive.load_chip(cache_dir, rec["city"], rec["chip_id"])
        for cname, cfg in configs.items():
            for level in levels:
                boxes, removed = select_boxes(arrays, detector, method, level, cfg)
                for iou in iou_thresholds:
                    res = match_boxes(boxes, refs, iou)
                    extent = arrays.get("extent_m")
                    if chance_reps and extent is not None:
                        seed = zlib.crc32(f"{rec['chip_id']}|{level}|{iou}".encode())
                        cm = chance_matches(boxes, refs, extent, iou, chance_reps, seed)
                    else:
                        cm = float("nan")
                    rows.append({
                        "chip_id": rec["chip_id"], "city": rec["city"], "block": rec["block"],
                        "split": rec["split"], "detector": detector, "method": method,
                        "config": cname, "level": level, "iou": iou,
                        "matches": res.matches, "predictions": res.predictions,
                        "references": res.references, "removed": removed,
                        "chance_matches": cm})
    return pd.DataFrame(rows)


def pooled_table(df: pd.DataFrame, by: list) -> pd.DataFrame:
    cols = ["matches", "predictions", "references", "removed"]
    if "chance_matches" in df:
        cols.append("chance_matches")
    g = df.groupby(by)[cols].sum(min_count=1).reset_index()
    g["precision"], g["recall"], g["f1"] = prf(g["matches"], g["predictions"], g["references"])
    if "chance_matches" in g:
        g["chance_f1"] = prf(g["chance_matches"], g["predictions"], g["references"])[2]
    return g


def chance_f1_by_level(df: pd.DataFrame, config: str, method: str, iou: float):
    """Pooled chance-level F1 per level, aligned with counts_tensor's levels."""
    sub = df[(df.config == config) & (df.method == method) & (df.iou == iou)]
    pooled = pooled_table(sub, ["level"]).sort_values("level")
    return pooled["level"].tolist(), pooled["chance_f1"].to_numpy()


def counts_tensor(df: pd.DataFrame, config: str, method: str, iou: float):
    """Return (levels, counts (L, chips, 3), block per chip) for the bootstrap."""
    sub = df[(df.config == config) & (df.method == method) & (df.iou == iou)]
    levels = sorted(sub.level.unique())
    chips = sorted(sub.chip_id.unique())
    arr = np.zeros((len(levels), len(chips), 3))
    for i, lv in enumerate(levels):
        part = sub[sub.level == lv].set_index("chip_id").loc[chips]
        arr[i] = part[["matches", "predictions", "references"]].to_numpy()
    blocks = sub.drop_duplicates("chip_id").set_index("chip_id").loc[chips, "block"].to_numpy()
    return levels, arr, blocks


def owl_grid(cfg: dict) -> dict:
    ev = cfg["evaluation"]
    grid = {}
    for (pname, idx), thr, mx, mn in itertools.product(
            ev["prompt_sets"].items(), ev["score_thresholds"],
            ev["max_size_m"], ev["min_area_m2"]):
        name = f"{pname}|t{thr:g}|max{mx:g}|min{mn:g}"
        grid[name] = EvalConfig(tuple(idx), float(thr), float(mx), float(mn))
    return grid


def tune(args) -> None:
    cfg = load_config(args.config)
    cache = Path(args.cache)
    index = pd.read_csv("archive/index.csv").to_dict("records")
    have = {p.stem for p in cache.rglob("*.npz")}
    records = [r for r in index if r["chip_id"] in have and r["split"] == "dev"]
    levels = cfg["sweep"]["levels_gsd_m"]
    grid = owl_grid(cfg)
    df = evaluate_records(records, cache, args.detector, args.method, levels, grid, [0.25])
    pooled = pooled_table(df, ["config", "level"])
    mean_f1 = pooled.groupby("config")["f1"].mean().sort_values(ascending=False)
    native = pooled[pooled.level == min(levels)].set_index("config")["f1"].sort_values(ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out, index=False)
    print("Best by mean F1 over levels:\n", mean_f1.head(5))
    print("Best at finest level only:\n", native.head(5))
    print("Config fields:", {k: asdict(v) for k, v in list(grid.items())[:1]})


def tune_colour(args) -> None:
    from .cache import colour_variants
    cfg = load_config(args.config)
    cache = Path(args.cache)
    index = pd.read_csv("archive/index.csv").to_dict("records")
    have = {p.stem for p in cache.rglob("*.npz")}
    records = [r for r in index if r["chip_id"] in have and r["split"] == "dev"]
    levels = cfg["sweep"]["levels_gsd_m"]
    grid = {f"max{m:g}": EvalConfig((), 0.0, float(m), 0.0) for m in cfg["evaluation"]["max_size_m"]}
    parts = []
    for name in colour_variants(cfg):
        df = evaluate_records(records, cache, f"colour-{name}", args.method, levels, grid, [0.25])
        df["config"] = name + "|" + df["config"]
        parts.append(pooled_table(df, ["config", "level"]))
    pooled = pd.concat(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out, index=False)
    mean_f1 = pooled.groupby("config")["f1"].mean().sort_values(ascending=False)
    print("Best colour settings by mean F1 over levels:\n", mean_f1.head(8))
    print("paper_fixed:\n", mean_f1[[c for c in mean_f1.index if c.startswith("paper_fixed")]])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("tune-colour")
    c.add_argument("--config", default="configs/sweep.yaml")
    c.add_argument("--cache", default="archive/dev")
    c.add_argument("--method", default="area")
    c.add_argument("--out", default="results/tune_colour.csv")
    t = sub.add_parser("tune")
    t.add_argument("--config", default="configs/sweep.yaml")
    t.add_argument("--cache", default="archive/dev")
    t.add_argument("--detector", default="owlvit")
    t.add_argument("--method", default="area")
    t.add_argument("--out", default="results/tune_owlvit.csv")
    args = ap.parse_args()
    if args.cmd == "tune":
        tune(args)
    elif args.cmd == "tune-colour":
        tune_colour(args)


if __name__ == "__main__":
    main()
