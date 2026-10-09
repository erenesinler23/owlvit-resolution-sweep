"""Shared detector interface.

Reconstructed from the paper's description: a detector takes three image
channels, an affine transform and a coordinate reference system, and returns
geometries with unique detection identifiers. Reconcile this file with the
original src/ package when the two repositories are compared.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from affine import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class Detection:
    det_id: str
    geometry: BaseGeometry
    score: Optional[float] = None


class Detector(ABC):
    @abstractmethod
    def detect(self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
               transform: Affine, raster_crs) -> list[Detection]:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def pixel_box_to_geometry(x0: float, y0: float, x1: float, y1: float,
                          transform: Affine) -> BaseGeometry:
    """Convert a pixel-space box (col/row) to a box in the transform's frame."""
    ax, ay = transform @ (x0, y0)
    bx, by = transform @ (x1, y1)
    return box(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
