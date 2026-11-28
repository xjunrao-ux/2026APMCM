"""Plot the fitted response probabilities and times at the requested distances."""

import json

from figure_data import ANALYSIS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from _utils.plot_utils import COLORS, PALETTE, save_fig, setup_style


DISTANCES = np.array([50, 100, 150])
SOUNDS = ("control", "flat", "mod", "tv")
LABELS = ("对照声", "flat", "mod", "tv")
MARKERS = ("D", "o", "s", "^")


def panel(ax, values, colors, title, ylabel, limits):
    offsets = np.array([-7.5, -2.5, 2.5, 7.5])
    ax.axvspan(129, 169, facecolor=PALETTE[0], alpha=.075, linewidth=0, zorder=0)
    ax.grid(axis="y", linestyle="--", alpha=.36, color=COLORS["grid"])
    ax.set_axisbelow(True)
    for i, (color, marker) in enumerate(zip(colors, MARKERS)):
        point, lo, hi = values[i].T
        locations = DISTANCES + offsets[i]
        ax.plot(locations, point, color=color, linewidth=1.0, alpha=.5, zorder=2)
        for j in np.flatnonzero(np.isfinite(point)):
            ax.errorbar(locations[j], point[j],
                        yerr=[[point[j]-lo[j]], [hi[j]-point[j]]], fmt=marker,
                        color=color, markersize=6.5, capsize=3, linewidth=1.45,
                        markeredgecolor="white", markeredgewidth=.8, zorder=3)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=11)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=7)
    ax.set_xlabel("回放距离（m）", fontsize=10, labelpad=6)
    ax.set_xticks(DISTANCES)
    ax.set_xlim(31, 169)
    ax.set_ylim(*limits)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_estimates(data):
    lookup = {(r["sound"], r["distance_m"]): r["outcomes"]["Any"] for r in data["rows"]}
    probability = np.full((len(SOUNDS), len(DISTANCES), 3), np.nan)
    latency = np.full_like(probability, np.nan)
    for i, sound in enumerate(SOUNDS):
        for j, distance in enumerate(DISTANCES):
            row = lookup[sound, distance]
            if row["reported_conditional_median_seconds"] is not None:
                probability[i, j] = row["probability_by_60"]
                latency[i, j] = row["reported_conditional_median_seconds"]
    assert set(zip(*np.where(~np.isfinite(latency[..., 0])))) == {(2, 1)}
    assert lookup["mod", 100]["distance_events"] == 0 and lookup["mod", 100]["distance_sample"] == 5
    assert all(lookup[sound, 150]["exact_sample"] == 0 for sound in SOUNDS)
    for values in (probability, latency):
        defined = values[np.isfinite(values[..., 0])]
        assert np.all((defined[:, 1] <= defined[:, 0]) & (defined[:, 0] <= defined[:, 2]))
    return probability, latency


def main():
    setup_style()
    path = ANALYSIS / "q3_latency_scenarios_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    probability, latency = read_estimates(data)
    colors = ("#62737E", PALETTE[0], PALETTE[2], "#226F79")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.15, 4.75))
    fig.subplots_adjust(left=.088, right=.975, top=.69, bottom=.24, wspace=.30)
    fig.text(.088, .952, "代表性距离下的企鹅首次响应", fontsize=17, weight="bold", va="top")
    fig.text(.088, .873, f"固定观测企鹅数量为 {data['group_size']:g} 只，按实测情境比例加权",
             fontsize=9.2, color=COLORS["text"], va="top")
    handles = [Line2D([0], [0], marker=m, color=c, linewidth=1, markersize=6,
                      markeredgecolor="white", label=label)
               for label, m, c in zip(LABELS, MARKERS, colors)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, .824),
               frameon=False, ncol=4, fontsize=9, handlelength=1.5, columnspacing=1.75)

    panel(left, probability, colors, "A   60 秒内是否出现响应",
          "响应概率", (0, .85))
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_yticks([0, .2, .4, .6, .8])
    panel(right, latency, colors, "B   有响应时多久开始",
          "中位潜伏期（秒）", (1.35, 3.42))
    right.set_yticks([1.5, 2, 2.5, 3])
    fig.text(.088, .112, "点与线段表示模型估计及逐项 95% 区间。mod 在 100 m 实测 0/5，两个面板均留空。",
             fontsize=8.1, color=COLORS["text"], va="center")
    fig.text(.088, .062, "浅色区为 150 m 的预测条件，该距离没有恰好 8 只的实测回放。",
             fontsize=8.1, color=COLORS["text"], va="center")
    out = ANALYSIS.parent / "figures" / "q3_latency_03_scenarios"
    save_fig(fig, str(out), formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    main()
