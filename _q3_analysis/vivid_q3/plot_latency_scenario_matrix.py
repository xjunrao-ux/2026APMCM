"""Annotated matrices for the representative response estimates."""

import json

from figure_data import ANALYSIS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from _utils.palette_maps import contrast_text
from _utils.plot_utils import save_fig, setup_style
from plot_latency_scenarios import LABELS, read_estimates


INK = "#172820"
MUTED = "#52625A"
PROBABILITY = ("#F3F7F4", "#AFCABA", "#6B927E")
LATENCY = ("#FAF7F1", "#E8CFB9", "#BD7D5B")


def draw_matrix(fig, left, estimates, stops, scale, title, percent=False):
    ax = fig.add_axes([left, .232, .356, .425])
    cax = fig.add_axes([left+.047, .163, .262, .015])
    color_map = LinearSegmentedColormap.from_list(title, stops)
    sns.heatmap(estimates[:, :, 0], cmap=color_map, vmin=scale[0], vmax=scale[1],
                xticklabels=("50 m", "100 m", "150 m*"), yticklabels=LABELS,
                linewidths=1.4, linecolor="white", ax=ax, cbar_ax=cax,
                cbar_kws={"orientation": "horizontal", "ticks": scale})
    fig.text(left, .753, title, ha="left", va="top", fontsize=13, weight="bold", color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, labelsize=9, pad=7, colors=MUTED)
    ax.set_yticklabels(LABELS, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    cax.tick_params(axis="x", length=0, pad=2, labelsize=7.5, colors=MUTED)
    if percent:
        cax.set_xticklabels([f"{n:.0%}" for n in scale])
    else:
        cax.set_xticklabels([f"{n:g}" for n in scale])
    surface = ax.collections[0]
    for row, col in np.ndindex(estimates.shape[:2]):
        center, lo, hi = estimates[row, col]
        if np.isnan(center):
            ax.add_patch(Rectangle((col+.045, row+.06), .91, .88, facecolor="#F2F4F1",
                                   edgecolor="#AAB9AE", linestyle=(0, (2.5, 2)), linewidth=.75))
            ax.text(col+.5, row+.41, "无响应", ha="center", va="center",
                    fontsize=10.5, weight="bold", color=MUTED)
            ax.text(col+.5, row+.67, "实测 0/5", ha="center", va="center", fontsize=8, color=MUTED)
            continue
        ink = contrast_text(surface.cmap(surface.norm(center)))
        main_text = f"{center:.1%}" if percent else f"{center:.2f}"
        interval = f"{100*lo:.1f}–{100*hi:.1f}%" if percent else f"{lo:.2f}–{hi:.2f}"
        ax.text(col+.5, row+.39, main_text, ha="center", va="center",
                fontsize=17, weight="bold", color=ink)
        ax.text(col+.5, row+.71, interval, ha="center", va="center",
                fontsize=8.1, color=ink, alpha=.94)


def main():
    setup_style()
    model_path = ANALYSIS / "q3_latency_scenarios_results.json"
    data = json.loads(model_path.read_text(encoding="utf-8"))
    p, time = read_estimates(data)
    fig = plt.figure(figsize=(9.4, 4.55), facecolor="white")
    fig.text(.045, .958, "不同回放距离下的首次响应",
             ha="left", va="top", fontsize=20, weight="bold", color=INK)
    fig.text(.047, .854, f"固定为 {data['group_size']:g} 只企鹅，按实测情境比例加权",
             ha="left", va="top", fontsize=9, color=MUTED)
    fig.add_artist(plt.Line2D([.047, .97], [.808, .808], transform=fig.transFigure,
                              color="#DCE6DF", linewidth=.7))
    draw_matrix(fig, .131, p, PROBABILITY, (0, .8), "60 秒内的响应概率", percent=True)
    draw_matrix(fig, .612, time, LATENCY, (1.8, 2.8), "有响应时的中位潜伏期（秒）")
    fig.text(.047, .068, "数值下方为逐项 95% 区间。* 150 m 没有恰好 8 只的实测回放，数值为模型预测。",
             ha="left", va="center", fontsize=8.1, color=MUTED)
    out = ANALYSIS.parent / "figures" / "q3_latency_04_scenario_matrix"
    save_fig(fig, str(out), formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    main()
