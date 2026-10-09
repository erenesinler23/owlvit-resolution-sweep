"""Per-city peak, break point and curve for the frozen settings.

Reads counts_<det>_<method>_iou<iou>_fixed.csv from one or more result folders (each row is
one chip, level and IoU), splits by city, and applies the same block bootstrap and break-point
rule as the main report. Writes results/per_city.csv.

    .venv/bin/python scripts/per_city.py --dirs results/test results/vegas
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from resolution_sweep.cache import load_config  # noqa: E402
from resolution_sweep.report import curve_rows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--iou", default="0.25")
ap.add_argument("--out", default="results/per_city.csv")
args = ap.parse_args()
cfg = load_config("configs/sweep.yaml")
bs = cfg["bootstrap"]
rows = []
for d in args.dirs:
    for det in ("owlvit", "owlv2", "colour"):
        for method in ("area", "blur_resize"):
            f = Path(d) / f"counts_{det}_{method}_iou{args.iou}_fixed.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            for city, g in df.groupby("city"):
                r = curve_rows(g, "fixed", method, float(args.iou), bs["n_boot"], bs["seed"], bs["alpha"])
                lv = dict(zip(r["levels"], r["f1"]))
                lo = dict(zip(r["levels"], r["lo"]))
                hi = dict(zip(r["levels"], r["hi"]))
                rows.append({"city": city, "detector": det, "method": method,
                             "n_chips": g.chip_id.nunique(),
                             "peak_gsd": r["peak_level"], "peak_f1": r["peak_f1"],
                             "break_gsd": r["break_point"],
                             "break_lo": r["break_ci"][0] if r["break_ci"] else None,
                             "break_hi": r["break_ci"][1] if r["break_ci"] else None,
                             "f1_0.9": lv.get(0.9), "f1_2.4": lv.get(2.4), "f1_2.4_lo": lo.get(2.4),
                             "f1_2.4_hi": hi.get(2.4), "f1_3.6": lv.get(3.6), "f1_5": lv.get(5.0)})
out = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(out.round(2).to_string(index=False))
