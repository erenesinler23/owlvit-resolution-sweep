"""Cluster bootstrap over spatial blocks.

Chips from one block are spatially correlated, so whole blocks are resampled.
Counts are (matches, predictions, references) per chip; pooled metrics come
from summed counts, as in the paper's per-location tables.
"""
from __future__ import annotations

import numpy as np

from .matching import prf


def _cluster_sums(counts: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """counts (..., n_chips, 3) -> (..., n_clusters, 3)."""
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    out = np.zeros(counts.shape[:-2] + (len(uniq), 3), dtype=float)
    for i in range(counts.shape[-2]):
        out[..., inv[i], :] += counts[..., i, :]
    return out


def _ci(x: np.ndarray, alpha: float):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(x, 100 * alpha / 2)),
            float(np.percentile(x, 100 * (1 - alpha / 2))))


def cluster_bootstrap(counts, clusters, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Return {metric: (point, lo, hi)} in percent for one set of chips."""
    counts = np.asarray(counts, dtype=float)
    sums = _cluster_sums(counts, clusters)
    point = prf(*sums.sum(axis=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sums.shape[0], size=(n_boot, sums.shape[0]))
    boot = sums[idx].sum(axis=1)
    bp, br, bf = prf(boot[:, 0], boot[:, 1], boot[:, 2])
    result = {}
    for name, pt, b in (("precision", point[0], bp), ("recall", point[1], br),
                        ("f1", point[2], bf)):
        lo, hi = _ci(b, alpha)
        result[name] = (float(pt), lo, hi)
    return result


def bootstrap_f1_curves(counts, clusters, n_boot: int = 2000, seed: int = 0):
    """Paired bootstrap of F1 across resolution levels.

    counts has shape (levels, chips, 3). The same block resample is applied to
    every level, so replicates keep the curve shape and break points can be
    computed per replicate. Returns (point (L,), boot (n_boot, L)) in percent.
    """
    counts = np.asarray(counts, dtype=float)
    sums = _cluster_sums(counts, clusters)            # (L, C, 3)
    point = prf(*sums.sum(axis=1).T)[2]               # (L,)
    rng = np.random.default_rng(seed)
    n_clusters = sums.shape[1]
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[:, idx, :].sum(axis=2)                # (L, n_boot, 3)
    f1 = prf(boot[..., 0], boot[..., 1], boot[..., 2])[2]
    return point, f1.T
