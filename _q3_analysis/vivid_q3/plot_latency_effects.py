"""Visualize the conditional time-position estimates and bootstrap intervals."""

import json

from figure_data import ANALYSIS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _utils.plot_utils import COLORS, PALETTE, save_fig, setup_style


TERMS = (
    ("距离翻倍", "log2_distance_over_80"),
    ("企鹅数量翻倍", "log2_group_size"),
    ("flat 相对对照", "flat_vs_control"),
    ("mod 相对对照", "mod_vs_control"),
    ("tv 相对对照", "tv_vs_control"),
    ("进食后相对赶路", "after_vs_commuting"),
    ("搜索食物相对赶路", "searching_vs_commuting"),
)
ENDPOINTS = (("任一行为首次出现", "Any"), ("首次鸣叫", "Vocal"))


def main():
    setup_style()
    model = json.loads((ANALYSIS / "q3_latency_model_results.json").read_text(encoding="utf-8"))
    scores = np.exp([[model["time"][key]["report_coefficients"][term] for _, key in ENDPOINTS]
                     for _, term in TERMS])
    assert np.all(scores[:, :, 1] <= scores[:, :, 0])
    assert np.all(scores[:, :, 0] <= scores[:, :, 2])
    fig, ax = plt.subplots(figsize=(7.9, 5.45))
    fig.subplots_adjust(left=.227, right=.955, top=.77, bottom=.22)
    fig.text(.075, .945, "哪些因素关联到更晚的首次响应？", fontsize=16, weight="bold", va="top")
    fig.text(.075, .875, "60 秒内已响应的回放 · 条件时间模型的逐项 95% 区间",
             fontsize=9, color=COLORS["text"], va="top")
    ax.grid(axis="x", alpha=.38, linestyle="-", color=COLORS["grid"])
    ax.set_axisbelow(True)
    ax.axvline(1, color=COLORS["ref_line"], linestyle="--", linewidth=1.1, zorder=2)
    ax.axhspan(-.5, .5, facecolor=PALETTE[0], alpha=.075, zorder=0)
    y_base = np.arange(len(TERMS))
    offsets = (-.17, .17)
    colors = (PALETTE[0], PALETTE[2])
    symbols = ("o", "s")
    for j, (name, _) in enumerate(ENDPOINTS):
        for i in range(len(TERMS)):
            point, lo, hi = scores[i, j]
            ax.errorbar(point, y_base[i] + offsets[j], xerr=[[point-lo], [hi-point]],
                        fmt=symbols[j], color=colors[j], markersize=6.6, capsize=3,
                        linewidth=1.5, markeredgecolor="white", markeredgewidth=.8,
                        label=name if i == 0 else "", zorder=3)
    ax.set_yticks(y_base, [name for name, _ in TERMS], fontsize=9.5)
    ax.set_ylim(len(TERMS)-.5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(.78, 1.40)
    ax.set_xticks([.8, .9, 1, 1.1, 1.2, 1.3, 1.4])
    ax.set_xticklabels(["0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4"])
    ax.set_xlabel("截断前的时间位置比（大于 1 表示更晚）", fontsize=10, labelpad=8)
    legend_elements = [
        Line2D([0], [0], marker=symbols[j], color="w", markerfacecolor=colors[j],
               markersize=7, label=name)
        for j, (name, _) in enumerate(ENDPOINTS)
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2,
              frameon=False, handlelength=1.1, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(.075, .063, "区间由 3,000 次录音轨道整组重抽样得到。抬头与靠近的选定时间模型仅含截距。",
             fontsize=8.2, color=COLORS["text"], va="center")
    out = ANALYSIS.parent / "figures" / "q3_latency_02_effects"
    save_fig(fig, str(out), formats=("pdf", "png", "svg"))
    np.set_printoptions(precision=4)
    print("time-position ratios by feature (Any, Vocal):")
    print(scores)


if __name__ == "__main__":
    main()
