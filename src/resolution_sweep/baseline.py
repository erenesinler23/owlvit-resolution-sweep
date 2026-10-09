"""Chance-level baseline: relocate every predicted box uniformly at random.

The number and size of predictions are kept, their position is destroyed. The
F1 this earns is what a detector with the same output shape scores by luck on
the same chips. It sets the floor for the break-point rule.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry.base import BaseGeometry

from .matching import match_boxes


def relocate_boxes(boxes: np.ndarray, extent: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x0 = rng.uniform(0.0, np.maximum(extent[0] - w, 0.0))
    y0 = rng.uniform(0.0, np.maximum(extent[1] - h, 0.0))
    return np.stack([x0, y0, x0 + w, y0 + h], axis=1)


def relocate_to_refs(boxes: np.ndarray, refs: Sequence[BaseGeometry],
                     rng: np.random.Generator) -> np.ndarray:
    """Move each box centre onto the centroid of a reference building.

    This is an oracle-location baseline. It knows where the buildings are and
    keeps only the number and size of the predictions. Centroids are drawn
    without replacement while they last, then with replacement.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cents = np.array([[g.centroid.x, g.centroid.y] for g in refs])
    n = len(boxes)
    idx = rng.permutation(len(cents))[:n]
    if n > len(cents):
        idx = np.concatenate([idx, rng.integers(0, len(cents), n - len(cents))])
    c = cents[idx]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([c[:, 0] - w / 2, c[:, 1] - h / 2, c[:, 0] + w / 2, c[:, 1] + h / 2], axis=1)


def chance_matches(boxes: np.ndarray, refs: Sequence[BaseGeometry], extent: Sequence[float],
                   iou_threshold: float, n_rep: int = 20, seed: int = 0,
                   mode: str = "uniform") -> float:
    """Mean number of matches over n_rep random relocations of the boxes.

    mode "uniform" places boxes anywhere on the chip. mode "oracle" places them
    on random reference building centres.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0 or len(refs) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    if mode == "oracle":
        moved = lambda: relocate_to_refs(boxes, refs, rng)  # noqa: E731
    else:
        moved = lambda: relocate_boxes(boxes, extent, rng)  # noqa: E731
    counts = [match_boxes(moved(), refs, iou_threshold).matches for _ in range(n_rep)]
    return float(np.mean(counts))
