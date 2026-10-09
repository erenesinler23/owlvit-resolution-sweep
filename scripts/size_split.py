"""Does the break point move with building size? Split chips by their median building size.

Chips with at least `--min-refs` reference buildings are ranked by the median equivalent side
(square root of footprint area) of their buildings and cut into terciles, pooled over all
cities. Frozen-setting F1 curves and break points are computed per tercile with the same block
bootstrap as the main report. If the break point follows building size in pixels, the
large-building tercile should break at a coarser GSD.

    .venv/bin/python scripts/size_split.py --dirs results/test results/vegas \
        --caches archive/test archive/vegas
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

sys.path.insert(0, "src")
from resolution_sweep.cache import load_config  # noqa: E402
from resolution_sweep.report import curve_rows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--caches", nargs="+", required=True)
ap.add_argument("--iou", default="0.25")
ap.add_argument("--min-refs", type=int, default=5)
ap.add_argument("--out", default="results/size_split.csv")
args = ap.parse_args()
cfg = load_config("configs/sweep.yaml")
bs = cfg["bootstrap"]

med = {}
for cache in args.caches:
    for f in Path(cache).rglob("*.refs.json"):
        refs = [Polygon(r[0], r[1:]) for r in json.loads(f.read_text())]
        if len(refs) >= args.min_refs:
            med[f.name.replace(".refs.json", "")] = float(np.median([np.sqrt(p.area) for p in refs]))
s = pd.Series(med)
cuts = s.quantile([1 / 3, 2 / 3]).to_numpy()
group = pd.cut(s, [-np.inf, cuts[0], cuts[1], np.inf], labels=["small", "medium", "large"])
print("chips", len(s), "tercile cuts (m)", cuts.round(2))
print(s.groupby(group, observed=True).agg(["count", "median"]).round(2))

rows = []
for det in ("owlvit", "owlv2"):
    for method in ("area", "blur_resize"):
        df = pd.concat([pd.read_csv(Path(d) / f"counts_{det}_{method}_iou{args.iou}_fixed.csv")
                        for d in args.dirs if (Path(d) / f"counts_{det}_{method}_iou{args.iou}_fixed.csv").exists()])
        df = df[df.chip_id.isin(s.index)]
        for g in ("small", "medium", "large"):
            sub = df[df.chip_id.map(group) == g]
            r = curve_rows(sub, "fixed", method, float(args.iou), bs["n_boot"], bs["seed"], bs["alpha"])
            lv = dict(zip(r["levels"], r["f1"]))
            rows.append({"detector": det, "method": method, "group": g,
                         "n_chips": sub.chip_id.nunique(),
                         "median_side_m": float(s[group == g].median()),
                         "peak_gsd": r["peak_level"], "peak_f1": r["peak_f1"],
                         "break_gsd": r["break_point"],
                         "break_lo": r["break_ci"][0] if r["break_ci"] else None,
                         "break_hi": r["break_ci"][1] if r["break_ci"] else None,
                         "f1_1.8": lv.get(1.8), "f1_2.4": lv.get(2.4), "f1_3.6": lv.get(3.6)})
            print(rows[-1], flush=True)
out = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(out.round(2).to_string(index=False))
