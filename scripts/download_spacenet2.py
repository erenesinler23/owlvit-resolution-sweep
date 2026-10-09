"""Download and extract the SpaceNet 2 Paris and Khartoum training tarballs (about 10 GB).

Public bucket, unsigned access. Skips a file whose local size matches the remote size.
Usage: python scripts/download_spacenet2.py [--out data/spacenet2]
"""
import argparse
import tarfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "spacenet-dataset"
PREFIX = "spacenet/SN2_buildings/tarballs/"
FILES = ["SN2_buildings_train_AOI_3_Paris.tar.gz", "SN2_buildings_train_AOI_5_Khartoum.tar.gz",
         "SN2_buildings_train_AOI_2_Vegas.tar.gz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/spacenet2")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for name in FILES:
        key = PREFIX + name
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        dest = out / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"skip {name} (already complete)")
        else:
            print(f"download {name} ({size / 1e9:.1f} GB)")
            s3.download_file(BUCKET, key, str(dest))
        target = out / name.replace(".tar.gz", "")
        if target.exists():
            print(f"skip extract {name}")
            continue
        target.mkdir()
        with tarfile.open(dest) as tf:
            tf.extractall(target, filter="data")
        print(f"extracted to {target}")


if __name__ == "__main__":
    main()
