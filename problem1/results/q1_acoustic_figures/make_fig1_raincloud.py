"""Fig. 1 閳?Raincloud distributions of the eight acoustic measurements.

Figure contract
---------------
Results-level question
    How do the eight acoustic measurements distribute within each of the three
    sea-surface call types, and where do the distributions overlap?
Figure-level claim
    Every spectral measurement carries a low-frequency component in mod and tv
    calls that flat calls largely lack, whereas call duration and pulse rate
    overlap almost completely across types.
Archetype
    quantitative grid (2 rows x 4 columns of raincloud panels), one panel per
    measurement; no panel is the hero because the compared quantities are units-
    incompatible and the claim is about the shape of eight distributions.
Panels
    a call duration      b first fundamental F01    c second fundamental F02
    d peak frequency     e Q25 energy frequency     f Q50 energy frequency
    g Q75 energy frequency   h pulse rate
Encoding per group
    half violin = kernel density (Gaussian, Scott bandwidth x 0.35)
    narrow box  = median, interquartile range, 1.5 x IQR whiskers
    right dots  = every individual call (no row excluded)
Data
    APMCM_C_Preprocessing_Q1/Data01_clean.csv, 108 calls, flat 44 / mod 40 / tv 24.
    F02 is only defined for two-voice calls and pulse rate only for mod and tv;
    these are structural gaps and are shown as such, never imputed.
Run: python make_fig1_raincloud.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    COL, LABEL, NEUTRAL_DARK, NEUTRAL_MID, OUT, SHADE, TYPE_ORDER,
    apply_style, add_panel_label, raincloud, require_matplotlib_panel_alignment, type_axis,
)
from q1_stats_core import (
    describe, groups, kruskal_family, load, pairwise_family, pulse_rate,
    second_fundamental, threshold_share,
)

# MANDATORY font + editable-text settings (kept in each figure source)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

apply_style()

df = load()
n_by_type = {t: int((df["CallType"] == t).sum()) for t in TYPE_ORDER}

# ---- source-data tables (descriptive statistics and tests used in all figures) ----
describe(df).to_csv(OUT / "fig1_metrics_by_type.csv", index=False, encoding="utf-8-sig")
kruskal_family(df).assign(test="Kruskal-Wallis H, Holm over 6 metrics").to_csv(
    OUT / "fig1_omnibus_tests.csv", index=False, encoding="utf-8-sig")
pairwise_family(df).assign(test="Mann-Whitney U, two-sided, Holm over 18 contrasts").to_csv(
    OUT / "fig1_pairwise_tests.csv", index=False, encoding="utf-8-sig")
pulse = pulse_rate(df)
f02 = second_fundamental(df)
pd.DataFrame([
    {"comparison": "mod/tv", "metric": "PulseRate", "n_mod": pulse["n_mod"], "n_tv": pulse["n_tv"],
     "median_mod": pulse["median_mod"], "median_tv": pulse["median_tv"], "U": pulse["U"],
     "p": pulse["p_raw"], "cliff_delta": pulse["cliff_delta"],
     "note": "single pre-specified comparison; not in the Holm family"},
    {"comparison": "tv: F02 vs F01", "metric": "F02", "n_mod": np.nan, "n_tv": f02["n"],
     "median_mod": np.nan, "median_tv": f02["median_f02"], "U": f02["wilcoxon_W"],
     "p": f02["wilcoxon_p"], "cliff_delta": np.nan,
     "note": (f"F02 above F01 in {f02['n_f02_above_f01']}/{f02['n']} calls; "
              f"median difference {f02['median_diff']:.1f} Hz")},
]).to_csv(OUT / "fig1_supplementary_tests.csv", index=False, encoding="utf-8-sig")

PANELS = [
    ("Duration", "Call duration (s)", None),
    ("F01", "F01 (Hz)", None),
    ("F02", "F02 (Hz)", ["tv"]),
    ("Fpeak", "Peak frequency (Hz)", None),
    ("Q25", "Q25 energy (Hz)", None),
    ("Q50", "Q50 energy (Hz)", None),
    ("Q75", "Q75 energy (Hz)", None),
    ("PulseRate", "Pulse rate (pulses/s)", ["mod", "tv"]),
]

fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 116 / 25.4))
low = threshold_share(df, "Fpeak", 500.0)

for i, (ax, (metric, ylabel, only)) in enumerate(zip(axes.ravel(), PANELS)):
    types = [t for t in TYPE_ORDER if only is None or t in only]
    values = {t: groups(df, metric, t) for t in types}
    span = np.concatenate([v for v in values.values() if v.size])
    lo, hi = float(span.min()), float(span.max())
    pad = 0.10 * (hi - lo)
    if metric == "Fpeak":
        ax.set_ylim(lo - 0.95 * pad, hi + pad)
    else:
        ax.set_ylim(lo - 0.35 * pad, hi + pad)
    ax.set_xlim(-0.5, 2.5)

    if metric == "Fpeak":
        ax.axhspan(ax.get_ylim()[0], 500, color=SHADE, alpha=0.55, lw=0, zorder=0)
        ax.axhline(500, color=NEUTRAL_MID, lw=0.5, ls=(0, (3, 2)), zorder=1)

    for t, pos in zip(TYPE_ORDER, (0, 1, 2)):
        if t not in values:
            ax.text(pos, 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1]), "not applicable",
                    fontsize=5.9, color="#9A9A9A", ha="center", va="center", rotation=90, rotation_mode="anchor")
            continue
        raincloud(ax, values[t], pos, COL[t])

    if metric == "Fpeak":
        for pos, t in zip((0, 1, 2), TYPE_ORDER):
            k, n = low[t]
            ax.text(pos, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"{k}/{n}", fontsize=6.2, color=COL[t], ha="center", va="bottom",
                    fontweight="bold")
        ax.set_yticks([200, 400, 500, 600, 800, 1000, 1200])

    if metric == "F02":
        ax.text(0.02, 0.98, "F02 > F01 in 24/24", transform=ax.transAxes, fontsize=5.9,
                color=COL["tv"], ha="left", va="top")

    if metric == "PulseRate":
        ax.text(0.02, 0.98, "mod vs tv: p = 0.28", transform=ax.transAxes, fontsize=5.9,
                color=NEUTRAL_DARK, ha="left", va="top")

    type_axis(ax, counts=n_by_type)
    if only is not None:
        ax.set_xticklabels([f"{LABEL[t]}\nn = {n_by_type[t]}" if t in types
                            else f"{LABEL[t]}\nn/a" for t in TYPE_ORDER])
    if metric == "PulseRate":
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"flat\nn = {n_by_type['flat']}",
                            f"mod\nn = {groups(df, 'PulseRate', 'mod').size}",
                            f"tv\nn = {groups(df, 'PulseRate', 'tv').size}"])
    ax.set_ylabel(ylabel, labelpad=1.5)
    add_panel_label(ax, "abcdefgh"[i])

fig.tight_layout(pad=0.35)
fig.canvas.draw()

require_matplotlib_panel_alignment(
    fig,
    json_out=str(OUT / "fig1_raincloud.alignment.json"),
    overlay_svg=str(OUT / "fig1_raincloud.alignment.svg"),
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
)

fig.savefig(OUT / "fig1_raincloud.svg")
fig.savefig(OUT / "fig1_raincloud.pdf")
fig.savefig(OUT / "fig1_raincloud.png", dpi=600)
fig.savefig(OUT / "fig1_raincloud.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("fig1 done")



