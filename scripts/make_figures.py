"""Figures for the paper from the saved result CSVs.

fig1: F1 vs GSD, one panel per detector, frozen setting (solid, CI band), per-level setting
      (dashed), uniform chance (dotted), oracle-location baseline for the frozen setting (grey).
fig2: real Sentinel-2 at 10 m against simulated 10 m, if results exist.

    .venv/bin/python scripts/make_figures.py --dir results/test --out docs/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAMES = {"owlvit": "OWL-ViT", "owlv2": "OWLv2", "colour": "Colour rule"}
COL = {"fixed": "#1f77b4", "per_level": "#d62728", "chance": "#7f7f7f", "oracle": "#2ca02c"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/test")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--method", default="area")
    ap.add_argument("--iou", default="0.25")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    orc = pd.read_csv(d / f"oracle_diff_{args.method}_iou{args.iou}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, det in zip(axes, NAMES):
        for kind in ("fixed", "per_level"):
            c = pd.read_csv(d / f"curve_{det}_{args.method}_iou{args.iou}_{kind}.csv")
            if kind == "per_level" and det == "colour":
                continue
            ax.plot(c.levels, c.f1, color=COL[kind], ls="-" if kind == "fixed" else "--",
                    marker="o", ms=3.5, lw=1.8,
                    label="frozen setting" if kind == "fixed" else "re-tuned per resolution")
            ax.fill_between(c.levels, c.lo, c.hi, color=COL[kind], alpha=0.15, lw=0)
            if kind == "fixed":
                ax.plot(c.levels, c.chance, color=COL["chance"], ls=":", lw=1.6,
                        label="chance (random positions)")
        o = orc[(orc.detector == det) & (orc.kind == "fixed")].sort_values("level")
        ax.plot(o.level, o.oracle_f1, color=COL["oracle"], ls="-.", lw=1.5,
                label="oracle locations (frozen boxes)")
        ax.set_xscale("log")
        ax.set_xticks([0.3, 0.6, 1.2, 2.4, 5, 10])
        ax.set_xticklabels(["0.3", "0.6", "1.2", "2.4", "5", "10"])
        ax.set_xlabel("ground sampling distance (m)")
        ax.set_title(NAMES[det])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"F1 (%), IoU {args.iou}")
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"fig1_f1_vs_gsd{args.tag}.png", dpi=200)
    print("wrote", out / f"fig1_f1_vs_gsd{args.tag}.png")


def fig2(out: Path, method: str = "area", iou: str = "0.25") -> None:
    """Left: frozen F1 vs GSD per city. Right: real Sentinel-2 against simulated 10 m, re-tuned settings."""
    import numpy as np
    cities = {"AOI_3_Paris": "Paris", "AOI_5_Khartoum": "Khartoum", "AOI_2_Vegas": "Las Vegas"}
    colours = {"AOI_3_Paris": "#1f77b4", "AOI_5_Khartoum": "#ff7f0e", "AOI_2_Vegas": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, det in zip(axes[:2], ("owlvit", "owlv2")):
        frames = [pd.read_csv(f) for f in (Path("results/test") / f"counts_{det}_{method}_iou{iou}_fixed.csv",
                                           Path("results/vegas") / f"counts_{det}_{method}_iou{iou}_fixed.csv")
                  if f.exists()]
        df = pd.concat(frames)
        for city, g in df.groupby("city"):
            p = g.groupby("level")[["matches", "predictions", "references"]].sum()
            f1 = 200 * p.matches / (p.predictions + p.references)
            ax.plot(p.index, f1, marker="o", ms=3.5, lw=1.7, color=colours[city], label=cities[city])
        ax.set_xscale("log")
        ax.set_xticks([0.3, 0.6, 1.2, 2.4, 5, 10])
        ax.set_xticklabels(["0.3", "0.6", "1.2", "2.4", "5", "10"])
        ax.set_xlabel("ground sampling distance (m)")
        ax.set_title(f"{NAMES[det]}, frozen setting, by city")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"F1 (%), IoU {iou}")
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[2]
    groups = [("Paris + Khartoum", Path("results/test_sentinel2/sentinel2.csv")),
              ("Las Vegas", Path("results/vegas_sentinel2/sentinel2.csv"))]
    xs, labels, w = [], [], 0.2
    pos = 0
    for gname, path in groups:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[(d.iou == float(iou)) & (d.setting == "per_level") & (d.detector.isin(["owlvit", "owlv2"]))]
        for det in ("owlvit", "owlv2"):
            r_area = d[(d.detector == det) & (d.sim_method == "area")].iloc[0]
            r_blur = d[(d.detector == det) & (d.sim_method == "blur_resize")].iloc[0]
            vals = [r_area.sim_f1, r_blur.sim_f1, r_area.s2_f1, r_area.oracle_f1_on_s2]
            cols = ["#9ecae1", "#6baed6", "#d62728", "#2ca02c"]
            for k, (v, c) in enumerate(zip(vals, cols)):
                ax.bar(pos + (k - 1.5) * w, v, w, color=c)
            xs.append(pos)
            labels.append(f"{NAMES[det]}\n{gname}")
            pos += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title("10 m: real Sentinel-2 vs simulation")
    ax.set_ylabel("F1 (%), re-tuned for 10 m")
    ax.grid(alpha=0.25, axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9ecae1", label="simulated, area"),
                       Patch(color="#6baed6", label="simulated, blur+resize"),
                       Patch(color="#d62728", label="real Sentinel-2"),
                       Patch(color="#2ca02c", label="oracle on real crops")], fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_cities_and_sentinel2.png", dpi=200)
    print("wrote", out / "fig2_cities_and_sentinel2.png")


if __name__ == "__main__":
    main()
    fig2(Path("docs/figures"))
