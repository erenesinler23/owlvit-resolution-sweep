"""Controlled resolution degradation.

Each chip is reduced to a target ground sampling distance (GSD), then enlarged
back to its original pixel grid with one fixed method. Detector boxes therefore
stay in native coordinates and are scored against the original labels.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

METHODS = ("area", "blur_resize")
UPSAMPLE = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}


def target_size(native_px: int, native_gsd: float, target_gsd: float) -> int:
    """Pixels needed to cover the same ground extent at target_gsd (round half up)."""
    return max(1, int(math.floor(native_px * native_gsd / target_gsd + 0.5 + 1e-9)))


def effective_gsd(native_px: int, native_gsd: float, target_gsd: float) -> float:
    """Actual GSD after integer rounding of the downsampled size."""
    return native_px * native_gsd / target_size(native_px, native_gsd, target_gsd)


def degrade(rgb: np.ndarray, native_gsd: float, target_gsd: float,
            method: str = "area", up: str = "cubic",
            blur_sigma_factor: float = 0.5) -> np.ndarray:
    """Return an image of the original shape that carries only target_gsd detail."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if up not in UPSAMPLE:
        raise ValueError(f"up must be one of {tuple(UPSAMPLE)}")
    if rgb.ndim != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected a uint8 HxWxC array")
    if target_gsd <= native_gsd + 1e-9:
        return rgb.copy()

    h, w = rgb.shape[:2]
    factor = target_gsd / native_gsd
    th = target_size(h, native_gsd, target_gsd)
    tw = target_size(w, native_gsd, target_gsd)

    if method == "area":
        small = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        sigma = blur_sigma_factor * factor
        ksize = 2 * int(math.ceil(3 * sigma)) + 1
        blurred = cv2.GaussianBlur(rgb, (ksize, ksize), sigma)
        small = cv2.resize(blurred, (tw, th), interpolation=cv2.INTER_LINEAR)

    big = cv2.resize(small, (w, h), interpolation=UPSAMPLE[up])
    return np.clip(big, 0, 255).astype(np.uint8)
