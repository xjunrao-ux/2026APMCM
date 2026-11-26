"""Shared style and drawing helpers for the Q1 acoustic figures.

Python (matplotlib) is the exclusive plotting backend for all three figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

OUT = HERE

COL = {"flat": "#0F4D92", "mod": "#42949E", "tv": "#9A4D8E"}
LABEL = {"flat": "flat", "mod": "mod", "tv": "tv"}
TYPE_ORDER = ("flat", "mod", "tv")
NEUTRAL_DARK = "#3A3A3A"
NEUTRAL_MID = "#7A7A7A"
NEUTRAL_LIGHT = "#C8C8C8"
SHADE = "#E3E1DC"
SIGNAL = "#B64342"

RNG = np.random.default_rng(20261125)


def apply_style() -> None:
    mpl.use("Agg")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def tint(hex_color: str, weight: float = 0.86) -> tuple[float, float, float]:
    c = np.array(mpl.colors.to_rgb(hex_color))
    return tuple((1 - weight) * c + weight * np.ones(3))


def add_panel_label(ax, label, x=0.0, y=1.0, dx=-5.0, dy=3.0, fontsize=8):
    offset = ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        ha="left",
        va="bottom",
    )


def _not_applicable(ax, pos, text="not applicable", y=None):
    ymin, ymax = ax.get_ylim()
    ax.text(pos, ymin + 0.5 * (ymax - ymin) if y is None else y, text, fontsize=6.0,
            color=NEUTRAL_MID, ha="center", va="center", rotation=90, rotation_mode="anchor")


def raincloud(ax, values, pos, color, violin_w=0.30, jitter=0.13, ms=2.6,
              bw_factor=0.35, show_box=True, shade_alpha=0.55):
    """Half violin (left), narrow box (centre) and jittered observations (right)."""
    values = np.asarray(values, float)
    if values.size < 2:
        return
    lo, hi = float(values.min()), float(values.max())
    y0, y1 = ax.get_ylim()
    grid = np.linspace(lo, hi, 240)
    try:
        dens = gaussian_kde(values, bw_method=bw_factor)(grid)
    except Exception:
        dens = np.zeros_like(grid)
    dens = dens / dens.max() * violin_w if dens.max() > 0 else dens
    ax.fill_betweenx(grid, pos - 0.10 - dens, pos - 0.10, color=color, alpha=shade_alpha,
                     lw=0.0, zorder=2)
    ax.plot(pos - 0.10 - dens, grid, color=color, lw=0.6, zorder=3)
    ax.plot([pos - 0.10, pos - 0.10], [max(lo, y0), min(hi, y1)], color=color, lw=0.6,
            alpha=0.9, zorder=3)
    if show_box:
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        wlo = values[values >= q1 - 1.5 * iqr].min()
        whi = values[values <= q3 + 1.5 * iqr].max()
        ax.add_patch(
            mpl.patches.Rectangle((pos - 0.045, q1), 0.09, max(q3 - q1, 1e-9),
                                  facecolor=tint(color, 0.7), edgecolor=color, lw=0.7, zorder=4)
        )
        ax.plot([pos - 0.045, pos + 0.045], [med, med], color=color, lw=1.1, zorder=5)
        ax.plot([pos, pos], [wlo, q1], color=color, lw=0.7, zorder=3)
        ax.plot([pos, pos], [q3, whi], color=color, lw=0.7, zorder=3)
    xs = pos + 0.13 + RNG.uniform(0, 2 * jitter, values.size)
    ax.scatter(xs, values, s=ms, color=color, alpha=0.55, linewidths=0, zorder=3)


def type_axis(ax, positions=(0, 1, 2), counts=None, fontsize=6.4):
    ax.set_xticks(list(positions))
    if counts is None:
        ax.set_xticklabels([LABEL[t] for t in TYPE_ORDER])
    else:
        ax.set_xticklabels([f"{LABEL[t]}\nn = {counts[t]}" for t in TYPE_ORDER])
