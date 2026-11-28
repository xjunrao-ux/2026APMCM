"""Compare the cumulative and conditional first-response curves."""

import json

import numpy as np
from figure_data import ANALYSIS
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from _fit_q3_work import CTX, D, N, design, prob
from q3_latency_model import conditional_cdf, time_parameters
from _utils.plot_utils import COLORS, PALETTE, save_fig, setup_style, smart_labels


def standardized_curves(model, times):
    occ = model["occurrence"]
    binary, combo = (np.asarray(occ[k]) for k in ("binary_coefficients", "combination_coefficients"))
    time_fit = model["time"]["Any"]
    coefficients = np.asarray(time_fit["coefficients"])
    columns = np.asarray(time_fit["features"])
    curves = []
    for sound in range(4):
        x = design(sound, CTX, D, N)[:, :8]
        p = 1 - prob(x, (binary, combo))[:, 0]
        mu, sigma = time_parameters(x[:, columns], coefficients)
        curve = (p[:, None] * conditional_cdf(times, mu[:, None], sigma)).mean(axis=0)
        np.testing.assert_allclose(curve[-1], p.mean())
        curves.append(curve)
    cumulative = np.stack([curves[0], np.mean(curves[1:], axis=0)])
    conditional = cumulative / cumulative[:, -1, None]
    assert np.all(np.diff(cumulative, axis=1) >= -1e-12)
    return cumulative, conditional


def curve_panel(ax, time, methods, stop, ylim, ylabel, title):
    colors = (PALETTE[2], PALETTE[0])
    ax.grid(True, linestyle="--", alpha=0.32, color=COLORS["grid"])
    ax.set_axisbelow(True)
    for i, (name, vals) in enumerate(methods.items()):
        ax.plot(time, vals, color=colors[i], linewidth=2.5 if i else 1.5,
                linestyle="-" if i else "--", label=name, alpha=1.0 if i else 0.83, zorder=3)
    ours = list(methods.values())[1]
    ax.fill_between(time, ours, alpha=0.10, color=colors[1], linewidth=0, zorder=1)
    ax.set_xlabel("回放开始后的时间（秒）", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(0, stop)
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return colors


def main():
    setup_style()
    model = json.loads((ANALYSIS / "q3_latency_model_results.json").read_text(encoding="utf-8"))
    times = np.linspace(0, 60, 2401)
    cumulative, conditional = standardized_curves(model, times)
    medians = [np.interp(0.5, curve, times) for curve in conditional]
    names = ("对照声", "自然声（三类等权）")
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10.0, 4.25))
    fig.subplots_adjust(left=.09, right=.971, bottom=.22, top=.76, wspace=.32)
    fig.text(.09, .945, "企鹅首次响应的概率与等待时间", fontsize=17, weight="bold", va="top")
    fig.text(.09, .853, "对照声与自然声代入相同的 278 组实测情境、距离和企鹅数量",
             fontsize=9, color=COLORS["text"], va="top")
    colors = curve_panel(ax, times, dict(zip(names, cumulative)), 69, .73,
                         "已响应的回放比例", "A   全部回放")
    end = cumulative[:, -1]
    smart_labels(ax, [60, 60], end, [f"{p:.1%}" for p in end], colors=colors,
                 fontsize=9, offset=(6, 0))
    at_five = np.interp(5, times, cumulative[1])
    ax.scatter([5], [at_five], color=colors[1], s=37, edgecolors="white", zorder=5)
    ax.annotate(f"5 秒内  {at_five:.1%}", xy=(5, at_five), xytext=(13, .36),
                fontsize=9, color=colors[1],
                arrowprops=dict(arrowstyle="->", color=colors[1], lw=1.1), zorder=5)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")

    curve_panel(zoom, times, dict(zip(names, conditional)), 10, 1.04,
                "已响应回放中的累计比例", "B   60 秒内有响应的回放")
    zoom.axhline(0.5, color=COLORS["ref_line"], linestyle=":", linewidth=.9, zorder=2)
    zoom.scatter(medians, [.5, .5], color=colors, s=38, edgecolors="white", zorder=5)
    zoom.annotate(f"中位等待时间\n自然声  {medians[1]:.2f} 秒\n对照声  {medians[0]:.2f} 秒",
                  xy=(medians[1], .5), xytext=(4.75, .145), fontsize=8.8,
                  color=COLORS["text"],
                  arrowprops=dict(arrowstyle="->", color=colors[1], lw=1.1), zorder=5)
    zoom.set_xticks([0, 2, 4, 6, 8, 10])
    fig.text(.09, .07, "曲线为模型点估计。B 图放大前 10 秒，分母为 60 秒内出现任一行为的回放。",
             color=COLORS["text"], fontsize=8, va="center")
    out = ANALYSIS.parent / "figures" / "q3_latency_01_curves"
    save_fig(fig, str(out), formats=("pdf", "png", "svg"))
    print(json.dumps({
        "seconds": [3, 5, 10, 60],
        "order": list(names),
        "cumulative": [[np.interp(t, times, c) for t in (3, 5, 10, 60)] for c in cumulative],
        "conditional": [[np.interp(t, times, c) for t in (3, 5, 10, 60)] for c in conditional],
        "conditional_medians_seconds": medians,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
