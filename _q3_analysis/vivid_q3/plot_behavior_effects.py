"""Behavior probability differences with matched uncertainty intervals."""
from figure_data import OUT, inputs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS


def panel(ax, labels, values, limits, ticks):
    # Vivid offset point intervals. The three marginal outcomes do not form a pooled estimate.
    ax.grid(axis='x', alpha=.32, linestyle='-', color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.axvline(0, color=COLORS['ref_line'], linewidth=.8, linestyle='--', zorder=1)
    y_base = np.arange(len(labels))
    offsets = np.linspace(-.25, .25, values.shape[1])
    colors = (PALETTE[0], PALETTE[2], PALETTE[5])
    markers = ('o', 's', '^')
    for j, (color, marker) in enumerate(zip(colors, markers)):
        for i, (point, low, high) in enumerate(values[:, j]):
            assert low <= point <= high
            ax.errorbar(point, y_base[i] + offsets[j], xerr=[[point-low], [high-point]],
                        fmt=marker, color=color, markersize=5.6, capsize=2.1,
                        linewidth=1.15, markeredgecolor='white', markeredgewidth=.75, zorder=3)
    ax.set(yticks=y_base, yticklabels=labels, xlim=limits, xticks=ticks, ylim=(len(labels)-.45, -.55))
    ax.set_xlabel('概率差（百分点）', fontsize=8.7)
    ax.tick_params(axis='both', labelsize=8)
    ax.tick_params(axis='y', length=0, pad=5)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    return [Line2D([0], [0], marker=m, color='white', markerfacecolor=c,
                   markersize=6, label=b) for c, m, b in zip(colors, markers, ('抬头', '鸣叫', '靠近'))]


def main():
    setup_style()
    _, _, factors = inputs()
    main_results = factors['additive']['contrasts']
    sound = 100 * np.asarray(main_results['sound'])
    covariates = 100 * np.concatenate((
        np.asarray(main_results['context'])[:, 1],
        np.asarray(main_results['distance'])[:, 1],
        np.asarray(main_results['size'])[:, 1],
    ))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.45))
    fig.subplots_adjust(left=.155, right=.98, bottom=.25, top=.84, wspace=.75)
    left.set_title('a   声型相对对照', loc='left', fontsize=10, fontweight='bold', pad=10)
    right.set_title('b   自然声下的条件比较', loc='left', fontsize=10, fontweight='bold', pad=10)
    handles = panel(left, ('flat', 'mod', 'tv', '自然声等权'), sound, (-4, 63), (0, 15, 30, 45, 60))
    panel(right, ('搜索食物−赶路', '进食后−赶路', '100 m−50 m', '8 只−4 只'),
          covariates, (-19, 14), (-15, -10, -5, 0, 5, 10))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3,
               fontsize=8.5, frameon=False, columnspacing=2, borderaxespad=0)
    fig.text(.063, .048, '点为主模型估计。线段为录音轨道重抽样的 95% 同时置信区间。两幅图的横轴刻度不同。', fontsize=7.6)
    save_fig(fig, str(OUT / '02_behavior_effects.pdf'), formats=('pdf', 'png', 'svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
