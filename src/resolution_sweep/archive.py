"""Replayable experiment archives.

One .npz per chip holds raw candidate boxes and scores for every
(detector, resize method, level). One .refs.json per chip holds the reference
polygons in the metric frame. A manifest records config, package versions and
file checksums. Evaluation reads only these files, never the dataset.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon


def level_tag(gsd: float) -> str:
    return f"g{gsd:g}".replace(".", "p")


def array_key(detector: str, method: str, gsd: float, field: str) -> str:
    return f"{detector}__{method}__{level_tag(gsd)}__{field}"


def chip_paths(root: Path, city: str, chip_id: str) -> tuple[Path, Path]:
    d = Path(root) / city
    return d / f"{chip_id}.npz", d / f"{chip_id}.refs.json"


def save_chip(root: Path, city: str, chip_id: str, arrays: dict, refs: list) -> None:
    npz, refs_path = chip_paths(root, city, chip_id)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    payload = [[list(map(list, p.exterior.coords))] +
               [list(map(list, r.coords)) for r in p.interiors] for p in refs]
    refs_path.write_text(json.dumps(payload))


def load_chip(root: Path, city: str, chip_id: str) -> tuple[dict, list]:
    npz, refs_path = chip_paths(root, city, chip_id)
    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    refs = [Polygon(rings[0], rings[1:]) for rings in json.loads(refs_path.read_text())]
    return arrays, refs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(root: Path, config: dict, extra: dict | None = None) -> Path:
    versions = {}
    for pkg in ("numpy", "scipy", "shapely", "opencv-python-headless", "rasterio",
                "torch", "transformers", "pillow", "pandas"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    files = sorted(p for p in Path(root).rglob("*") if p.suffix in (".npz", ".json")
                   and p.name != "manifest.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "geos": shapely.geos_version_string,
        "packages": versions,
        "config": config,
        "extra": extra or {},
        "checksums": {str(p.relative_to(root)): sha256_file(p) for p in files},
    }
    out = Path(root) / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out
