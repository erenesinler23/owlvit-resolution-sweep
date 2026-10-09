"""One-to-one IoU matching and metrics, following the paper's definitions.

A valid pair needs IoU >= threshold (and IoU > 0). Each prediction and each
reference joins at most one pair. The assignment first maximises the number of
valid pairs, then their total IoU. Coordinates are in metres, so size filters
are physical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely import STRtree
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class MatchResult:
    matches: int
    predictions: int
    references: int
    pairs: tuple  # (pred_index, ref_index, iou)


def size_mask(boxes: np.ndarray, max_size_m: Optional[float] = None,
              min_area_m2: float = 0.0) -> np.ndarray:
    """True for boxes kept. Boxes wider or taller than max_size_m are removed."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w * h) >= min_area_m2
    if max_size_m is not None:
        keep &= (w <= max_size_m) & (h <= max_size_m)
    return keep


def iou_matrix(preds: Sequence[BaseGeometry], refs: Sequence[BaseGeometry]) -> np.ndarray:
    n, m = len(preds), len(refs)
    iou = np.zeros((n, m), dtype=float)
    if n == 0 or m == 0:
        return iou
    p = np.asarray(preds, dtype=object)
    r = np.asarray(refs, dtype=object)
    tree = STRtree(r)
    pi, ri = tree.query(p, predicate="intersects")
    if len(pi) == 0:
        return iou
    inter = shapely.area(shapely.intersection(p[pi], r[ri]))
    union = shapely.area(p[pi]) + shapely.area(r[ri]) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(union > 0, inter / union, 0.0)
    iou[pi, ri] = vals
    return iou


def match(preds: Sequence[BaseGeometry], refs: Sequence[BaseGeometry],
          iou_threshold: float) -> MatchResult:
    n, m = len(preds), len(refs)
    if n == 0 or m == 0:
        return MatchResult(0, n, m, ())
    iou = iou_matrix(preds, refs)
    valid = (iou >= iou_threshold) & (iou > 0)
    if not valid.any():
        return MatchResult(0, n, m, ())
    # K exceeds any possible total IoU, so pair count dominates, IoU breaks ties.
    k = min(n, m) + 1.0
    weights = np.where(valid, k + iou, 0.0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    pairs = tuple((int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if valid[i, j])
    return MatchResult(len(pairs), n, m, pairs)


def match_boxes(pred_boxes: np.ndarray, refs: Sequence[BaseGeometry],
                iou_threshold: float) -> MatchResult:
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    preds = list(shapely.box(pred_boxes[:, 0], pred_boxes[:, 1],
                             pred_boxes[:, 2], pred_boxes[:, 3]))
    return match(preds, refs, iou_threshold)


def prf(matches, predictions, references):
    """Precision, recall and F1 in percent, NaN where the denominator is empty."""
    m, p, r = (np.asarray(x, dtype=float) for x in (matches, predictions, references))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(p > 0, 100.0 * m / p, np.nan)
        recall = np.where(r > 0, 100.0 * m / r, np.nan)
        f1 = np.where(p + r > 0, 200.0 * m / (p + r), np.nan)
    return precision, recall, f1
