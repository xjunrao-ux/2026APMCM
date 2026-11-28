"""Shared drawing and bootstrap inputs for the English figure set."""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

from matplotlib.colors import to_rgb
from figure_data import ANALYSIS


OUT_EN = ANALYSIS.parent / "figures" / "q3_english"
INK = "#263238"
TEAL = "#2B665D"
BLUE = "#4F667F"
RUST = "#94513C"
MUTED = "#596971"
GRAY = "#A8B0B3"


def configure_english():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelcolor": INK,
        "axes.unicode_minus": True,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    OUT_EN.mkdir(parents=True, exist_ok=True)


def bootstrap_results():
    with np.load(ANALYSIS / "behavior_plot_draws.npz") as archive:
        draws = {key: 100 * archive[key] for key in archive.files}
    assert draws["combination"].shape == (3000, 8)
    assert draws["sound"].shape == (3000, 4, 3)
    assert all(np.isfinite(array).all() for array in draws.values())
    np.testing.assert_allclose(draws["combination"].sum(axis=1), 0, atol=1e-10)
    return draws


def blend_with_white(color, amount=.78):
    return tuple(value * (1 - amount) + amount for value in to_rgb(color))


def draw_boxes(ax, values, positions, colors, width):
    box = ax.boxplot(values, positions=positions, widths=width, patch_artist=True,
                     whis=(2.5, 97.5), showfliers=False, manage_ticks=False,
                     medianprops={"linewidth": 1.1}, boxprops={"linewidth": 1},
                     whiskerprops={"linewidth": .85}, capprops={"linewidth": .85},
                     zorder=3)
    for i, color in enumerate(colors):
        box["boxes"][i].set_facecolor(blend_with_white(color))
        box["boxes"][i].set_edgecolor(color)
        box["medians"][i].set_color(color)
        for item in box["whiskers"][2*i:2*i+2] + box["caps"][2*i:2*i+2]:
            item.set_color(color)
    ax.axhline(0, color=MUTED, linewidth=.75, linestyle=(0, (4, 3)), zorder=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8ECED", alpha=.9, linewidth=.6)
    ax.tick_params(axis="both", labelsize=7.8, length=3)
