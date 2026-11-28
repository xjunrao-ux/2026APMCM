"""Vertical boxplots of bootstrap differences for individual behaviors."""
from figure_english import (
    BLUE, INK, MUTED, OUT_EN, RUST, TEAL, blend_with_white,
    bootstrap_results, configure_english, draw_boxes,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
from _utils.plot_utils import setup_style, save_fig


SERIES = (TEAL, RUST, BLUE)


def panel(ax, values, labels, ticks):
    x = np.arange(len(labels))
    positions = (x[:, None] + np.array([-.23, 0, .23])).ravel()
    colors = list(SERIES) * len(labels)
    draw_boxes(ax, values.reshape(len(values), -1), positions, colors, width=.185)
    ax.set_xticks(x, labels)
    ax.set_xlim(-.53, len(labels)-.45)
    low, high = np.percentile(values, [2.5, 97.5], axis=0)
    lower = min(0, low.min())
    upper = max(0, high.max())
    ax.set_ylim(lower - .10*(upper-lower), upper + .10*(upper-lower))
    ax.yaxis.set_major_locator(MultipleLocator(ticks))
    ax.tick_params(axis='x', length=0, pad=6)


def main():
    setup_style(language='en')
    configure_english()
    draws = bootstrap_results()
    conditional = np.concatenate(
        [draws[name][:, :, 1, :] for name in ('context', 'distance', 'size')], axis=1
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.43))
    fig.subplots_adjust(left=.116, right=.98, bottom=.265, top=.81, wspace=.30)
    a.set_title('a   Call type vs. control', loc='left', fontsize=9.6, fontweight='bold', pad=8)
    b.set_title('b   Conditions within natural calls', loc='left', fontsize=9.6, fontweight='bold', pad=8)
    panel(a, draws['sound'], ('Flat', 'Modulated', 'Two-voice', 'All calls'), 15)
    panel(b, conditional, ('Searching\nvs. transit', 'After feeding\nvs. transit',
                           '100 m\nvs. 50 m', '8 vs. 4\npenguins'), 5)
    a.set_ylabel('Adjusted probability difference (pp)', fontsize=8.4, labelpad=6, color=INK)
    handles = [Patch(facecolor=blend_with_white(c), edgecolor=c, label=s)
               for c, s in zip(SERIES, ('Head raising', 'Vocalizing', 'Approach'))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .985),
               ncol=3, fontsize=7.8, frameon=False, columnspacing=1.6)
    fig.text(.064, .09, '3,000 bootstrap refits by recording track. Boxes show the interquartile range and median.',
             fontsize=7.3, color=MUTED)
    fig.text(.064, .051, 'Whiskers mark the 2.5th and 97.5th percentiles. The panels have different vertical scales.',
             fontsize=7.3, color=MUTED)
    save_fig(fig, str(OUT_EN / '02_behavior_boxes.pdf'), formats=('pdf', 'png', 'svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
