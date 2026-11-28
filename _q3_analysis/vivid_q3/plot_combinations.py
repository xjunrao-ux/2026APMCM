"""Forest plot of joint behavioral outcomes from the adjusted probability table."""
from figure_data import COMBINATION_LABELS, COMBINATION_ORDER, OUT, inputs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS


def main():
    setup_style()
    _, results, _ = inputs()
    values = np.array([results['natural_equal', name] for name in COMBINATION_ORDER])
    control, natural, coefs, ci_lo, ci_hi = values.T
    reported = (ci_lo > 0) | (ci_hi < 0)
    assert np.array_equal(np.where(reported)[0], [0, 2, 4, 7])
    fig, ax = plt.subplots(figsize=(7.2, 3.88))
    fig.subplots_adjust(left=.29, right=.675, top=.91, bottom=.18)
    y = np.arange(len(COMBINATION_LABELS)) * 1.4
    text_coords = blended_transform_factory(fig.transFigure, ax.transData)
    for i in range(len(y)):
        if i % 2 == 0:
            ax.axhspan(y[i] - .58, y[i] + .58, xmin=-.59, xmax=1.79,
                       color=COLORS['bg_box'], zorder=0, clip_on=False)
    ax.axvline(0, color=COLORS['down'], linestyle='--', linewidth=1.05, alpha=.85, zorder=1)
    positive = PALETTE[0]
    lower = PALETTE[2]
    for i, (c, lo, hi, label, supported) in enumerate(zip(coefs, ci_lo, ci_hi, COMBINATION_LABELS, reported)):
        color = (positive if c > 0 else lower) if supported else COLORS['text']
        ax.plot([lo, hi], [y[i], y[i]], color=color, linewidth=1.20, zorder=2, solid_capstyle='round')
        ax.plot([lo, lo], [y[i]-.11, y[i]+.11], color=color, linewidth=.90)
        ax.plot([hi, hi], [y[i]-.11, y[i]+.11], color=color, linewidth=.90)
        ax.plot(c, y[i], 'o', color=color, markersize=5.1,
                markerfacecolor=color if supported else 'white', markeredgecolor=color,
                markeredgewidth=1.0, zorder=3)

        ax.text(.072, y[i], label, transform=text_coords, ha='left', va='center',
                fontsize=8.8, color=COLORS['text'], weight='bold' if supported else 'normal')
        for x, value in ((.752, control[i]), (.853, natural[i])):
            ax.text(x, y[i], f'{value:.2f}', transform=text_coords, ha='center', va='center',
                    fontsize=8.3, color=COLORS['text'])
        estimate = f'{c:+.2f}'.replace('-', '−')
        ax.text(.954, y[i], estimate, transform=text_coords, ha='center', va='center',
                fontsize=8.6, color=color, weight='bold' if supported else 'normal')

    for x, title, align in ((.072, '行为组合', 'left'), (.752, '对照声 %', 'center'),
                            (.853, '自然声 %', 'center'), (.954, '差值', 'center')):
        ax.text(x, -.99, title, transform=text_coords, ha=align, va='center', fontsize=8.7,
                color=COLORS['dark'], weight='bold')
    ax.set(yticks=[], xlim=(-64, 30), ylim=(10.44, -1.43),
           xticks=[-60, -40, -20, 0, 20])
    ax.set_xlabel('自然声－对照声（百分点）', fontsize=8.9)
    ax.tick_params(axis='x', labelsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.34, linestyle='-', color=COLORS['grid'])
    fig.text(.064, .062, '自然声按三类等权。实心点的 95% 同时置信区间排除零。组合均指同一次回放。', fontsize=7.6)
    save_fig(fig, str(OUT / '03_combinations.pdf'), formats=('pdf', 'png', 'svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
