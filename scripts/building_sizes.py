"""Reference building statistics per city: count, median equivalent side, share below 5 m, empty chips.

    .venv/bin/python scripts/building_sizes.py --caches archive/test archive/vegas
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ap = argparse.ArgumentParser()
ap.add_argument("--caches", nargs="+", required=True)
ap.add_argument("--out", default="results/building_sizes.csv")
args = ap.parse_args()
rows = []
for cache in args.caches:
    for city_dir in sorted(Path(cache).iterdir()):
        if not city_dir.is_dir():
            continue
        sides, empty, n_chips = [], 0, 0
        for f in city_dir.glob("*.refs.json"):
            n_chips += 1
            refs = [Polygon(r[0], r[1:]) for r in json.loads(f.read_text())]
            empty += len(refs) == 0
            sides += [float(np.sqrt(p.area)) for p in refs if p.area > 0]
        s = np.array(sides)
        rows.append({"city": city_dir.name, "chips": n_chips, "empty_chips": empty,
                     "buildings": len(s), "median_side_m": np.median(s),
                     "p25_side_m": np.percentile(s, 25), "p75_side_m": np.percentile(s, 75),
                     "share_below_5m": float((s < 5).mean())})
df = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(df.round(2).to_string(index=False))
