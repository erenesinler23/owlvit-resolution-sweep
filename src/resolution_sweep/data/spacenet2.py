"""SpaceNet 2 reader (pan-sharpened RGB chips and GeoJSON building polygons).

Verified from public pages: chips are 650 x 650 pixels at 0.30 m, labels are
polygons. NOT yet verified on disk: folder layout, bit depth, CRS and band
order. The globs live in configs/sweep.yaml so they can be corrected after
the first look at the extracted archive (see docs/PLAN.md, Day 1).

All geometry is returned in a metric frame: x = column * gsd, y = row * gsd
(row increases downward). IoU is unchanged by this uniform scaling.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from affine import Affine
from shapely.geometry import shape


@dataclass(frozen=True)
class ChipRef:
    chip_id: str
    city: str
    image_path: Path
    label_path: Path


def index_city(root: Path, city: str, image_glob: str, label_glob: str) -> list[ChipRef]:
    """Pair images and labels by the 'img<number>' token in their file names."""
    def key(p: Path):
        m = re.search(r"(img\d+)", p.name)
        return m.group(1) if m else None

    images = {key(p): p for p in sorted(root.glob(image_glob.format(city=city))) if key(p)}
    labels = {key(p): p for p in sorted(root.glob(label_glob.format(city=city))) if key(p)}
    return [ChipRef(f"{city}_{k}", city, images[k], labels[k])
            for k in sorted(images.keys() & labels.keys())]


def to_uint8(arr: np.ndarray, scale_max: float | None) -> np.ndarray:
    """uint8 passes through. Other dtypes use one linear scale for all bands.

    A single constant keeps band ratios intact, which the colour rule depends on.
    Values above scale_max saturate at 255.
    """
    if arr.dtype == np.uint8:
        return arr
    if not scale_max or scale_max <= 0:
        raise ValueError(f"{arr.dtype} data needs dataset.scale_max in the config.")
    return np.clip(np.rint(arr.astype(np.float64) * (255.0 / scale_max)), 0, 255).astype(np.uint8)


def read_rgb(path: Path, scale_max: float | None = None):
    """Return r, g, b (uint8), the chip's geo transform and CRS."""
    import rasterio
    with rasterio.open(path) as ds:
        arr = to_uint8(ds.read([1, 2, 3]), scale_max)
        return arr[0], arr[1], arr[2], ds.transform, ds.crs


def chip_centre(transform: Affine, width: int, height: int) -> tuple[float, float]:
    x, y = transform @ (width / 2.0, height / 2.0)
    return float(x), float(y)


def read_refs(label_path: Path, geo_transform: Affine, gsd: float,
              width: int, height: int, min_area_m2: float = 0.5) -> list:
    """Building polygons in the metric frame, clipped to the chip extent."""
    inv = ~geo_transform

    def to_metric(coords: np.ndarray) -> np.ndarray:
        x, y = coords[:, 0], coords[:, 1]
        col = inv.a * x + inv.b * y + inv.c
        row = inv.d * x + inv.e * y + inv.f
        return np.stack([col * gsd, row * gsd], axis=1)

    with open(label_path) as fh:
        features = json.load(fh).get("features", [])
    extent = shapely.box(0, 0, width * gsd, height * gsd)
    out = []
    for feat in features:
        if not feat.get("geometry"):
            continue
        geom = shape(feat["geometry"])
        if geom.is_empty:
            continue
        geom = shapely.make_valid(shapely.transform(geom, to_metric, include_z=False))
        geom = shapely.intersection(geom, extent)
        for part in shapely.get_parts(geom):
            if part.geom_type == "Polygon" and part.area >= min_area_m2:
                out.append(part)
    return out
