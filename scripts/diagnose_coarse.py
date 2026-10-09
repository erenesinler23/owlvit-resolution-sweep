"""What do the detectors match at coarse GSD? Matched-reference size and density on the test cache."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from resolution_sweep import archive  # noqa: E402
from resolution_sweep.cache import load_config  # noqa: E402
from resolution_sweep.evaluate import select_boxes  # noqa: E402
from resolution_sweep.matching import match_boxes  # noqa: E402
from resolution_sweep.report import resolve  # noqa: E402

cfg = load_config("configs/sweep.yaml")
frozen = yaml.safe_load(open("configs/frozen.yaml"))["detectors"]
index = pd.read_csv("archive/index.csv")
recs = index[index.split == "test"].to_dict("records")
have = {p.stem for p in Path("archive/test").rglob("*.npz")}
recs = [r for r in recs if r["chip_id"] in have]
print(len(recs), "chips")

rows = []
for det, levels in [("owlv2", [1.2, 5.0, 10.0]), ("owlvit", [1.2, 3.6, 5.0, 10.0])]:
    pl = frozen[det]["per_level"]
    for lv in levels:
        name = pl[str(lv)] if str(lv) in pl else pl[lv]
        key, ec = resolve(det, name, cfg)
        m_area, all_area, pred_area, nbuild = [], [], [], []
        npred = nmatch = 0
        for r in recs:
            arrays, refs = archive.load_chip(Path("archive/test"), r["city"], r["chip_id"])
            boxes, _ = select_boxes(arrays, key, "area", lv, ec)
            res = match_boxes(boxes, refs, 0.25)
            npred += res.predictions
            nmatch += res.matches
            all_area += [g.area for g in refs]
            m_area += [refs[j].area for (_, j, _) in res.pairs]
            pred_area += [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            nbuild.append(len(refs))
        m = np.array(m_area)
        a = np.array(all_area)
        p = np.array(pred_area)
        print(f"{det} {lv} m [{name}] preds {npred} matches {nmatch}")
        print(f"  ref area m2 median all {np.median(a):.0f} matched {np.median(m):.0f}; "
              f"mean all {a.mean():.0f} matched {m.mean():.0f}; "
              f"frac matched >1000 m2 {np.mean(m > 1000):.2f} vs all {np.mean(a > 1000):.2f}; "
              f"pred median area {np.median(p):.0f}")
