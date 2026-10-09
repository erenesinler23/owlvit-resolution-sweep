"""Block-bootstrap CIs for detector F1 minus oracle-location F1, per detector, setting and level.

Reads results/<dir>/oracle_perchip_<method>_iou<iou>.csv written by oracle_baseline.py.
Usage: python scripts/oracle_diff.py --dir results/test [--method area --iou 0.25]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from resolution_sweep.matching import prf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--method", default="area")
ap.add_argument("--iou", default="0.25")
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()

df = pd.read_csv(Path(args.dir) / f"oracle_perchip_{args.method}_iou{args.iou}.csv")
rng = np.random.default_rng(1)
rows = []
for (det, kind, lv), g in df.groupby(["detector", "kind", "level"]):
    s = g.groupby("block")[["matches", "predictions", "references", "oracle_matches"]].sum()
    a = s.to_numpy(dtype=float)
    idx = rng.integers(0, len(a), size=(args.n_boot, len(a)))
    tot = a[idx].sum(axis=1)
    f_det = prf(tot[:, 0], tot[:, 1], tot[:, 2])[2]
    f_orc = prf(tot[:, 3], tot[:, 1], tot[:, 2])[2]
    d = f_det - f_orc
    d = d[np.isfinite(d)]
    p = a.sum(axis=0)
    fd, fo = prf(p[0], p[1], p[2])[2], prf(p[3], p[1], p[2])[2]
    rows.append({"detector": det, "kind": kind, "level": lv, "f1": fd, "oracle_f1": fo,
                 "diff": fd - fo, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
out = pd.DataFrame(rows).sort_values(["detector", "kind", "level"])
out.to_csv(Path(args.dir) / f"oracle_diff_{args.method}_iou{args.iou}.csv", index=False)
print(out.round(1).to_string(index=False))
