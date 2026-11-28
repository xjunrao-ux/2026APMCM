"""Vertical boxplots for the eight joint behavioral outcomes."""
from figure_english import (
    BLUE, INK, MUTED, OUT_EN, TEAL, bootstrap_results, configure_english, draw_boxes,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from _utils.plot_utils import setup_style, save_fig


# The bits in the fitted response code are head raising, vocalizing, and approach.
ORDER = (0, 4, 2, 1, 6, 5, 3, 7)
LABELS = ('No\nresponse', 'Head raising\nonly', 'Vocalizing\nonly', 'Approach\nonly',
          'Head raising\n+ vocalizing', 'Head raising\n+ approach', 'Vocalizing\n+ approach',
          'All three\nbehaviors')


def main():
    setup_style(language='en')
    configure_english()
    draws = bootstrap_results()
    distribution = draws['combination'][:, ORDER]
    estimates = draws['combination_point'][list(ORDER)]
    palette = [BLUE, MUTED, TEAL, MUTED, TEAL, MUTED, MUTED, TEAL]
    fig, ax = plt.subplots(figsize=(7.2, 3.72))
    fig.subplots_adjust(left=.123, right=.987, top=.86, bottom=.252)
    x = np.arange(len(LABELS))
    draw_boxes(ax, distribution, x, palette, width=.44)
    ax.plot(x, estimates, 'D', markerfacecolor='white', markeredgecolor=INK,
            markeredgewidth=.8, markersize=3.9, linestyle='none', zorder=4)
    ax.set_xticks(x, LABELS)
    ax.set_xlim(-.52, len(LABELS)-.48)
    ax.set_ylim(-63, 33)
    ax.set_yticks([-60, -40, -20, 0, 20])
    assert np.all(np.percentile(distribution, 2.5, axis=0) > -63)
    assert np.all(np.percentile(distribution, 97.5, axis=0) < 33)
    ax.tick_params(axis='x', labelsize=7.1, length=0, pad=6)
    ax.set_ylabel('Natural calls minus control (pp)', fontsize=8.5, labelpad=7, color=INK)
    ax.set_title('Joint behavioral outcomes', loc='left', fontsize=9.7, fontweight='bold', pad=10)
    diamond = Line2D([], [], marker='D', markersize=4, markeredgecolor=INK,
                     markerfacecolor='white', linewidth=0, label='Full-data estimate')
    ax.legend(handles=[diamond], loc='upper right', bbox_to_anchor=(1., 1.14),
              frameon=False, fontsize=7.5, borderaxespad=0)
    fig.text(.064, .09, '3,000 recording-track refits of the additive model. Boxes show the interquartile range and median.',
             fontsize=7.3, color=MUTED)
    fig.text(.064, .051, 'Whiskers mark the 2.5th and 97.5th percentiles. Natural call types are equally weighted.',
             fontsize=7.3, color=MUTED)
    save_fig(fig, str(OUT_EN / '03_combination_boxes.pdf'), formats=('pdf', 'png', 'svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
