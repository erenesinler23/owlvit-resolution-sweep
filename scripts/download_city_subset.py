"""Download a random subset of chips for one SpaceNet 2 city from the public per-file S3 layout.

The Las Vegas tarball is 25.6 GB, so this fetches only n chips (PS-RGB, 8-band MS and building
geojson) into the same folder layout the tarball extraction produces for Paris and Khartoum.
Chip choice is a seeded random sample of the file list, made before any model has seen the city.

    .venv/bin/python scripts/download_city_subset.py --city AOI_2_Vegas --n 300 --seed 20261009
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "spacenet-dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="AOI_2_Vegas")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", default="data/spacenet2")
    args = ap.parse_args()
    city = args.city
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=16))
    base = f"spacenet/SN2_buildings/train/{city}/"
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=base + "PS-RGB/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    ids = sorted(k.rsplit("_img", 1)[1].split(".")[0] for k in keys)
    pick = sorted(random.Random(args.seed).sample(ids, min(args.n, len(ids))), key=int)
    print(len(ids), "chips available,", len(pick), "chosen", flush=True)
    root = Path(args.out) / f"SN2_buildings_train_{city}" / f"{city}_Train"
    jobs = []
    for i in pick:
        jobs.append((f"{base}PS-RGB/SN2_buildings_train_{city}_PS-RGB_img{i}.tif",
                     root / "RGB-PanSharpen" / f"RGB-PanSharpen_{city}_img{i}.tif"))
        jobs.append((f"{base}MS/SN2_buildings_train_{city}_MS_img{i}.tif",
                     root / "MUL" / f"MUL_{city}_img{i}.tif"))
        jobs.append((f"{base}geojson_buildings/SN2_buildings_train_{city}_geojson_buildings_img{i}.geojson",
                     root / "geojson" / "buildings" / f"buildings_{city}_img{i}.geojson"))

    def get(job):
        key, dest = job
        if dest.exists() and dest.stat().st_size > 0:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(dest) + ".part")
        Path(str(dest) + ".part").rename(dest)
        return 1

    with ThreadPoolExecutor(12) as ex:
        done = sum(ex.map(get, jobs))
    print("downloaded", done, "files to", root, flush=True)


if __name__ == "__main__":
    main()
