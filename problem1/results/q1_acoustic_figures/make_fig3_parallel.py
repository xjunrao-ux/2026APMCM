"""Fig. 3 鈥?Parallel-coordinate profiles of the six shared acoustic measurements.

Figure contract
---------------
Results-level question
    Do the three call types occupy different positions in the six-dimensional
    acoustic space, and how much do individual calls overlap between types?
Figure-level claim
    Individual calls of different types overlap heavily, yet the three median
    profiles are separated in the same direction on every spectral axis: flat
    profiles run above mod, and mod above tv, from F01 through Q50 and converge
    again at Q75, while the temporal axis (duration) does not separate them.
Archetype
    single quantitative panel (multivariate profile).
Panels
    a  one line per call across the six z-scored measurements, coloured by call
       type, with the three group median profiles drawn on top.
Encoding
    thin line = one call; bold line with markers = the group median profile.
    z-score = (x - mean) / SD computed over all 108 calls of that measurement,
    so every axis is on the same dimensionless scale; raw units are named on
    each axis.
Data
    APMCM_C_Preprocessing_Q1/Data01_clean.csv, 108 calls, no row excluded.
    F02 (two-voice only) and pulse rate (mod and tv only) are excluded here
    because they are not defined for all three types.
Run: python make_fig3_parallel.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from figure_common import (
    COL, NEUTRAL_DARK, NEUTRAL_MID, OUT, TYPE_ORDER,
    apply_style, add_panel_label, require_matplotlib_panel_alignment,
)
from q1_stats_core import groups, load

# MANDATORY font + editable-text settings (kept in each figure source)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

apply_style()

AXES = [
    ("Duration", "Duration\n(s)"),
    ("F01", "F01\n(Hz)"),
    ("Fpeak", "Peak freq.\n(Hz)"),
    ("Q25", "Q25 energy\n(Hz)"),
    ("Q50", "Q50 energy\n(Hz)"),
    ("Q75", "Q75 energy\n(Hz)"),
]

df = load()
z = {}
for metric, _ in AXES:
    col = df[metric].to_numpy(float)
    z[metric] = (col - np.nanmean(col)) / np.nanstd(col, ddof=1)

xs = np.arange(len(AXES))
fig, ax = plt.subplots(figsize=(183 / 25.4, 100 / 25.4))

zall = np.column_stack([z[m] for m, _ in AXES])
y0, y1 = float(np.min(zall)), float(np.max(zall))
pad = 0.10 * (y1 - y0)
ax.set_ylim(y0 - pad, y1 + pad)

# axes guides
for x, (metric, _) in zip(xs, AXES):
    ax.plot([x, x], [y0 - pad, y1 + pad], color="#D8D8D8", lw=0.8, zorder=0)
ax.axhline(0, color=NEUTRAL_MID, lw=0.6, ls=(0, (4, 2)), zorder=0)

# one line per call
for t in TYPE_ORDER:
    idx = df["CallType"].to_numpy() == t
    mat = np.column_stack([z[m][idx] for m, _ in AXES])
    for row in mat:
        ax.plot(xs, row, color=COL[t], lw=0.5, alpha=0.17, zorder=1, solid_capstyle="round")

# group median profiles
med_rows = []
for t in TYPE_ORDER:
    med = np.array([np.median(z[m][df["CallType"].to_numpy() == t]) for m, _ in AXES])
    med_rows.append((t, med))
    ax.plot(xs, med, color=COL[t], lw=2.1, zorder=3, solid_capstyle="round")
    ax.scatter(xs, med, s=24, facecolors="white", edgecolors=COL[t], linewidths=1.1,
               zorder=4)

for x, (metric, label) in zip(xs, AXES):
    ax.text(x, 1.03, label, transform=ax.get_xaxis_transform(), fontsize=6.4,
            color=NEUTRAL_DARK, ha="center", va="bottom", linespacing=1.15)

ax.set_xticks([])
ax.set_xlim(-0.55, len(AXES) - 0.45)
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_ylabel("Standardized value (z)")
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)

for t in TYPE_ORDER:
    n = int((df["CallType"] == t).sum())
    ax.plot([], [], color=COL[t], lw=2.1, label=f"{t} (n = {n})")
fig.legend(loc="upper left", bbox_to_anchor=(0.072, 0.985), ncol=3, fontsize=6.4,
           handlelength=1.8, handletextpad=0.4, columnspacing=1.2, frameon=False)
add_panel_label(ax, "a")

fig.subplots_adjust(left=0.105, right=0.99, top=0.80, bottom=0.045)
fig.canvas.draw()

fig.savefig(OUT / "fig3_parallel.svg")
fig.savefig(OUT / "fig3_parallel.pdf")
fig.savefig(OUT / "fig3_parallel.png", dpi=600)
fig.savefig(OUT / "fig3_parallel.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("fig3 done")





