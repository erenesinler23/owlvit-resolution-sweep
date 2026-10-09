"""Colour rule from the paper: red > 120, red-green > 25, red-blue > 25.

Connected pixels are grouped and each region becomes one bounding box.
Matches the original NaiveColorDetector: strict inequalities, scipy's default
4-connectivity (confirmed by reading the original source) and no minimum
region size. Use connectivity=8 only as a deliberate variant.
"""
from __future__ import annotations

import numpy as np
from affine import Affine
from scipy import ndimage

from .base import Detection, Detector, pixel_box_to_geometry


class ColourRuleDetector(Detector):
    def __init__(self, red_min: int = 120, rg_min: int = 25, rb_min: int = 25,
                 connectivity: int = 4, min_pixels: int = 1):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.red_min = red_min
        self.rg_min = rg_min
        self.rb_min = rb_min
        self.connectivity = connectivity
        self.min_pixels = min_pixels

    @property
    def name(self) -> str:
        return f"colour_r{self.red_min}_g{self.rg_min}_b{self.rb_min}"

    def mask(self, r, g, b) -> np.ndarray:
        r16, g16, b16 = (np.asarray(x).astype(np.int16) for x in (r, g, b))
        return (r16 > self.red_min) & ((r16 - g16) > self.rg_min) & ((r16 - b16) > self.rb_min)

    def detect(self, r, g, b, transform: Affine, raster_crs=None) -> list[Detection]:
        structure = np.ones((3, 3), dtype=int) if self.connectivity == 8 else None
        labelled, _ = ndimage.label(self.mask(r, g, b), structure=structure)
        out: list[Detection] = []
        for i, sl in enumerate(ndimage.find_objects(labelled)):
            if sl is None:
                continue
            if self.min_pixels > 1 and int((labelled[sl] == i + 1).sum()) < self.min_pixels:
                continue
            rows, cols = sl
            geom = pixel_box_to_geometry(cols.start, rows.start, cols.stop, rows.stop, transform)
            out.append(Detection(det_id=f"{self.name}-{i}", geometry=geom))
        return out
