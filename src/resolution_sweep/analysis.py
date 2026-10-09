"""Break point and the single F1-versus-resolution figure."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def break_point(gsd: Sequence[float], f1: Sequence[float], frac: float = 0.5,
                floor: Optional[Sequence[float]] = None, ref: str = "finest") -> Optional[float]:
    """GSD where normalised F1 first falls to frac.

    Normalised F1 is (F1 - floor) / (F1_finest - floor_finest), where floor is the
    per-level chance F1 (zero when not given). Linear interpolation in log10(GSD).
    Returns None when the finest level does not rise above its floor (a floor
    result, not a break point) or the curve never falls that far.
    """
    gsd = np.asarray(gsd, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    fl = np.zeros_like(f1) if floor is None else np.asarray(floor, dtype=float)
    order = np.argsort(gsd)
    gsd, f1, fl = gsd[order], f1[order], fl[order]
    # ref="finest": scale by the finest level. ref="peak": scale by the best level above
    # chance and search only coarser levels. Peak is the better choice when the curve is
    # not monotone, which it is not here.
    i0 = 0 if ref == "finest" else int(np.nanargmax(f1 - fl))
    span = f1[i0] - fl[i0]
    if not np.isfinite(span) or span <= 0 or f1[i0] <= 0:
        return None
    s = (f1 - fl) / span
    for i in range(i0 + 1, len(s)):
        if np.isfinite(s[i]) and s[i] <= frac:
            lo, hi = s[i - 1], s[i]
            t = 0.0 if hi == lo else (lo - frac) / (lo - hi)
            lg = np.log10(gsd[i - 1]) + t * (np.log10(gsd[i]) - np.log10(gsd[i - 1]))
            return float(10 ** lg)
    return None


def above_chance(boot_f1_finest: Sequence[float], chance_f1_finest: float,
                 alpha: float = 0.05) -> bool:
    """True if the lower CI bound of finest-level F1 exceeds chance-level F1.

    boot_f1_finest comes from bootstrap_f1_curves (column for the finest level).
    The chance F1 is a point estimate averaged over random relocations.
    """
    x = np.asarray(boot_f1_finest, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return bool(np.percentile(x, 100 * alpha / 2) > chance_f1_finest)


def plot_f1_curves(curves: dict, out_path: str, title: str = "") -> None:
    """curves: label -> dict(gsd=[...], f1=[...], lo=[...], hi=[...])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for label, c in curves.items():
        g = np.asarray(c["gsd"], dtype=float)
        ax.plot(g, c["f1"], marker="o", label=label)
        if "lo" in c and "hi" in c:
            ax.fill_between(g, c["lo"], c["hi"], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Ground sampling distance (m per pixel)")
    ax.set_ylabel("F1 (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
