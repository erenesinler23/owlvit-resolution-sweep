"""Fetch real Sentinel-2 L2A crops over SpaceNet 2 chip footprints.

For each chip, finds a low-cloud Sentinel-2 L2A scene (earth-search STAC, public COGs on AWS,
2018 to 2021) that covers the chip, rejects it if the scene classification layer marks cloud,
shadow or no data inside the footprint, and resamples the red, green and blue bands onto a
20 by 20 grid over the exact chip footprint (bilinear). Saves uint16 digital numbers.

    .venv/bin/python scripts/fetch_sentinel2.py --cities AOI_3_Paris AOI_5_Khartoum --split test

No labels are read here. Output: data/sentinel2/<chip_id>.npy and data/sentinel2/scenes.csv.
"""
import argparse
import csv
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

STAC = "https://earth-search.aws.element84.com/v1/search"
BAD_SCL = (0, 1, 3, 8, 9, 10)     # no data, saturated, shadow, cloud medium, cloud high, cirrus
N = 20
_local = threading.local()


def stac_search(bbox, dt, cloud, limit=60):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": dt, "limit": limit,
            "query": {"eo:cloud_cover": {"lt": cloud}}, "sortby": [
                {"field": "properties.eo:cloud_cover", "direction": "asc"}]}
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["features"]


def open_cached(href):
    cache = getattr(_local, "ds", None)
    if cache is None:
        cache = _local.ds = {}
    if href not in cache:
        cache[href] = rasterio.open(href)
    return cache[href]


def crop(href, bounds_ll, resampling):
    ds = open_cached(href)
    w, s, e, n = transform_bounds("EPSG:4326", ds.crs, *bounds_ll)
    dst = np.zeros((N, N), dtype=np.uint16)
    reproject(rasterio.band(ds, 1), dst, dst_transform=from_bounds(w, s, e, n, N, N),
              dst_crs=ds.crs, resampling=resampling)
    return dst


def contains(feat, b):
    x0, y0, x1, y1 = feat["bbox"]
    return x0 <= b[0] and y0 <= b[1] and x1 >= b[2] and y1 >= b[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/sentinel2")
    ap.add_argument("--cloud", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = [r for r in csv.DictReader(open("archive/index.csv"))
             if r["city"] in args.cities and r["split"] == args.split]
    if args.limit:
        index = index[: args.limit]

    boxes = {}
    for r in index:
        with rasterio.open(r["image_path"]) as ds:
            b = ds.bounds
        boxes[r["chip_id"]] = (b.left, b.bottom, b.right, b.top)

    scenes = {}
    for city in args.cities:
        bs = [boxes[r["chip_id"]] for r in index if r["city"] == city]
        if not bs:
            continue
        bb = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
        feats = stac_search(bb, "2018-01-01T00:00:00Z/2021-12-31T23:59:59Z", args.cloud)
        scenes[city] = feats
        print(city, "candidate scenes", len(feats), flush=True)

    def work(r):
        cid = r["chip_id"]
        dest = out / f"{cid}.npy"
        if dest.exists():
            return cid, "cached", ""
        b = boxes[cid]
        for feat in scenes[r["city"]]:
            if not contains(feat, b):
                continue
            a = feat["assets"]
            try:
                scl = crop(a["scl"]["href"], b, Resampling.nearest)
                if np.isin(scl, BAD_SCL).any():
                    continue
                rgb = np.dstack([crop(a[k]["href"], b, Resampling.bilinear)
                                 for k in ("red", "green", "blue")])
            except Exception as exc:  # network or window errors: try the next scene
                print("skip", cid, feat["id"], exc, flush=True)
                continue
            if (rgb == 0).any():
                continue
            np.save(dest, rgb)
            return cid, "ok", feat["id"]
        return cid, "none", ""

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(work, index), 1):
            rows.append(res)
            if i % 50 == 0:
                print(i, "/", len(index), flush=True)
    tag = "_".join(c.split("_")[-1] for c in args.cities)
    with open(out / f"scenes_{args.split}_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chip_id", "status", "scene"])
        w.writerows(rows)
    from collections import Counter
    print(Counter(r[1] for r in rows))


if __name__ == "__main__":
    main()
