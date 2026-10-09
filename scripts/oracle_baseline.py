"""Oracle-location baseline on a cache: how much of the coarse-GSD score is just 'boxes on buildings'?

For each detector and level, takes the per-level and fixed settings, keeps the number and size of
the predicted boxes, drops them onto random reference building centres, and scores them.
A detector below this line gets nothing from the image beyond where buildings are and what size
boxes it emits. Writes results/<out>/oracle_baseline.csv.
Usage: python scripts/oracle_baseline.py --cache archive/test --split test --out results/test
"""
import argparse
import sys
import zlib
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, "src")
from resolution_sweep import archive  # noqa: E402
from resolution_sweep.baseline import chance_matches  # noqa: E402
from resolution_sweep.cache import load_config  # noqa: E402
from resolution_sweep.evaluate import select_boxes  # noqa: E402
from resolution_sweep.matching import match_boxes, prf  # noqa: E402
from resolution_sweep.report import resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--iou", type=float, default=0.25)
ap.add_argument("--method", default="area")
ap.add_argument("--cities", nargs="*")
args = ap.parse_args()

cfg = load_config("configs/sweep.yaml")
frozen = yaml.safe_load(open("configs/frozen.yaml"))["detectors"]
levels = cfg["sweep"]["levels_gsd_m"]
index = pd.read_csv("archive/index.csv")
have = {p.stem for p in Path(args.cache).rglob("*.npz")}
recs = [r for r in index.to_dict("records") if r["chip_id"] in have and r["split"] == args.split
        and (not args.cities or r["city"] in args.cities)]
print(len(recs), "chips")

rows = []
for rec in recs:
    arrays, refs = archive.load_chip(Path(args.cache), rec["city"], rec["chip_id"])
    for det in ["owlvit", "owlv2", "colour"]:
        fz = frozen[det]
        settings = {"fixed": fz["fixed"]}
        for lv in levels:
            pl = fz["per_level"]
            settings[f"per_level@{lv}"] = pl[str(lv)] if str(lv) in pl else pl[lv]
        for sname, name in settings.items():
            if sname.startswith("per_level@"):
                lv_only = float(sname.split("@")[1])
                lvs = [lv_only]
            else:
                lvs = levels
            key, ec = resolve(det, name, cfg)
            for lv in lvs:
                boxes, _ = select_boxes(arrays, key, args.method, lv, ec)
                res = match_boxes(boxes, refs, args.iou)
                seed = zlib.crc32(f"{rec['chip_id']}|{lv}|{det}".encode())
                om = chance_matches(boxes, refs, arrays["extent_m"], args.iou, args.reps, seed, "oracle")
                rows.append({"chip_id": rec["chip_id"], "block": rec["block"], "city": rec["city"],
                             "detector": det, "kind": sname.split("@")[0], "level": lv,
                             "matches": res.matches, "predictions": res.predictions,
                             "references": res.references, "oracle_matches": om})

df = pd.DataFrame(rows)
Path(args.out).mkdir(parents=True, exist_ok=True)
df.to_csv(Path(args.out) / f"oracle_perchip_{args.method}_iou{args.iou}.csv", index=False)
g = df.groupby(["detector", "kind", "level"])[["matches", "predictions", "references", "oracle_matches"]].sum().reset_index()
g["f1"] = prf(g.matches, g.predictions, g.references)[2]
g["oracle_f1"] = prf(g.oracle_matches, g.predictions, g.references)[2]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
g.to_csv(out / "oracle_baseline.csv", index=False)
print(g.round(1).to_string())
