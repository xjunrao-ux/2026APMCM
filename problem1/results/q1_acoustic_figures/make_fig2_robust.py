"""Fig. 2 閳?Robust location, robust spread and robust pairwise shifts.

Figure contract
---------------
Results-level question
    How large are the between-type differences when location and spread are
    estimated with robust statistics instead of means and standard deviations?
Figure-level claim
    Robustly estimated, flat calls sit highest and tv calls lowest on every
    spectral measurement, the shift is at most ~6% of the pooled median, and
    only the flat-to-mod shift in peak frequency survives multiplicity control.
    Call duration and pulse rate show negligible robust shifts.
Archetype
    quantitative grid (1 row x 2 columns), one row per measurement shared by
    both panels so the same vertical position compares the same measurement.
Panels
    a  median and +/-1 MAD per call type, expressed as a percentage of the
       measurement's pooled median (common scale across mixed units).
    b  Hodges-Lehmann shift estimator with 95% percentile bootstrap CI for each
       pairwise contrast, same percentage scale.
Robust statistics
    median (location), MAD (spread), Hodges-Lehmann (shift). No mean, SD or
    t-test is used anywhere in this figure.
Data
    APMCM_C_Preprocessing_Q1/Data01_clean.csv, 108 calls. F02 and pulse-rate
    cells that are structurally absent are left empty, never imputed.
Run: python make_fig2_robust.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    COL, NEUTRAL_DARK, NEUTRAL_MID, OUT, TYPE_ORDER,
    apply_style, add_panel_label, require_matplotlib_panel_alignment,
)
from q1_stats_core import groups, kruskal_family, load, pairwise_family, pulse_rate

# MANDATORY font + editable-text settings (kept in each figure source)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

apply_style()

METRIC_ROWS = [
    ("Duration", "Duration"),
    ("F01", "F01"),
    ("Fpeak", "Peak freq."),
    ("Q25", "Q25"),
    ("Q50", "Q50"),
    ("Q75", "Q75"),
    ("PulseRate", "Pulse rate"),
]
PAIR_ROWS = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
TYPE_OFF = {"flat": 0.26, "mod": 0.0, "tv": -0.26}
PAIR_OFF = {("flat", "mod"): 0.26, ("flat", "tv"): 0.0, ("mod", "tv"): -0.26}
PAIR_MARKER = {("flat", "mod"): "o", ("flat", "tv"): "s", ("mod", "tv"): "^"}

df = load()
kw = kruskal_family(df)
pw = pairwise_family(df)
pulse = pulse_rate(df)
pair_lookup = {(r["metric"], r["comparison"]): r for r in pw.to_dict("records")}
pooled_median = {}
n_before = len(df)
for m, _ in METRIC_ROWS:
    observed = df[m].dropna()  # structural gaps only: F02 is tv-only, pulse rate is mod/tv-only
    n_after = int(observed.size)
    pooled_median[m] = float(np.median(observed.to_numpy(float)))
    print(f"{m}: n_before={n_before} n_after={n_after} excluded={n_before - n_after}")
rows_y = {m: len(METRIC_ROWS) - 1 - i for i, (m, _) in enumerate(METRIC_ROWS)}


def hodges_lehmann(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.subtract.outer(a, b).ravel()))


def hl_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 20261125):
    rng = np.random.default_rng(seed)
    est = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, a.size, replace=True)
        sb = rng.choice(b, b.size, replace=True)
        est[i] = np.median(np.subtract.outer(sa, sb).ravel())
    return np.percentile(est, [2.5, 97.5])


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(183 / 25.4, 112 / 25.4), sharey=True)

# ---------------------------------------------------------------- panel a ----
loc_rows = []
for metric, label in METRIC_ROWS:
    denom = pooled_median[metric]
    for t in TYPE_ORDER:
        vals = groups(df, metric, t)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        rel, rel_mad = med / denom * 100, mad / denom * 100
        y = rows_y[metric] + TYPE_OFF[t]
        ax_a.plot([rel - rel_mad, rel + rel_mad], [y, y], color=COL[t], lw=1.9, alpha=0.40,
                  solid_capstyle="butt", zorder=2)
        ax_a.scatter([rel], [y], s=19, color=COL[t], linewidths=0, zorder=3)
        loc_rows.append({"metric": metric, "call_type": t, "n": int(vals.size), "median": med,
                         "mad": mad, "rel_median_pct": rel, "rel_mad_pct": rel_mad})

ax_a.axvline(100, color=NEUTRAL_MID, lw=0.6, ls=(0, (4, 2)), zorder=1)
ax_a.set_yticks([rows_y[m] for m, _ in METRIC_ROWS])
ax_a.set_yticklabels([label for _, label in METRIC_ROWS])
ax_a.set_ylim(-0.72, len(METRIC_ROWS) - 0.28)
lo_a = min(r["rel_median_pct"] - r["rel_mad_pct"] for r in loc_rows)
hi_a = max(r["rel_median_pct"] + r["rel_mad_pct"] for r in loc_rows)
ax_a.set_xlim(np.floor(lo_a - 2), np.ceil(hi_a + 2))
ax_a.set_xlabel("Median (dot) and \u00b11 MAD (bar), % of pooled median")
for t in TYPE_ORDER:
    ax_a.scatter([], [], s=17, color=COL[t], label=t)
ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3, fontsize=6.2,
            handletextpad=0.25, columnspacing=0.9, borderaxespad=0.1)
add_panel_label(ax_a, "a")

# ---------------------------------------------------------------- panel b ----
shift_rows = []
for metric, label in METRIC_ROWS:
    denom = pooled_median[metric]
    for a, b in PAIR_ROWS:
        if metric == "PulseRate" and (a, b) != ("mod", "tv"):
            continue
        va, vb = groups(df, metric, a), groups(df, metric, b)
        if va.size == 0 or vb.size == 0:
            continue
        hl = hodges_lehmann(va, vb)
        lo, hi = hl_boot(va, vb)
        p = pulse["p_raw"] if metric == "PulseRate" else pair_lookup[(metric, f"{a}/{b}")]["p_holm"]
        y = rows_y[metric] + PAIR_OFF[(a, b)]
        rel = hl / denom * 100
        ax_b.errorbar([rel], [y],
                      xerr=[[max(0.0, rel - lo / denom * 100)], [max(0.0, hi / denom * 100 - rel)]],
                      fmt="none", ecolor=NEUTRAL_MID, elinewidth=0.8, capsize=1.8, capthick=0.8,
                      alpha=0.75, zorder=2)
        ax_b.scatter([rel], [y], s=19, marker=PAIR_MARKER[(a, b)],
                     facecolors=NEUTRAL_DARK if p < 0.05 else "white",
                     edgecolors=NEUTRAL_DARK, linewidths=0.9, zorder=3)
        shift_rows.append({"metric": metric, "comparison": f"{a}/{b}", "n_a": int(va.size),
                           "n_b": int(vb.size), "hodges_lehmann": hl,
                           "hl_pct_of_pooled_median": rel, "ci_low_pct": lo / denom * 100,
                           "ci_high_pct": hi / denom * 100, "p_value": p})

ax_b.axvline(0, color=NEUTRAL_MID, lw=0.6, ls=(0, (4, 2)), zorder=1)
ax_b.set_ylim(-0.72, len(METRIC_ROWS) - 0.28)
lo_b = min(r["ci_low_pct"] for r in shift_rows)
hi_b = max(r["ci_high_pct"] for r in shift_rows)
ax_b.set_xlim(np.floor(lo_b - 1.5), np.ceil(hi_b + 1.5))
ax_b.set_xlabel("Hodges\u2013Lehmann shift, % of pooled median")
for a, b in PAIR_ROWS:
    ax_b.scatter([], [], s=19, marker=PAIR_MARKER[(a, b)], facecolors=NEUTRAL_DARK,
                 edgecolors=NEUTRAL_DARK, label=f"{a} \u2013 {b}")
ax_b.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3, fontsize=6.2,
            handletextpad=0.25, columnspacing=0.9, borderaxespad=0.1)
ax_b.text(0.995, 0.012, "solid: p < 0.05", transform=ax_b.transAxes, fontsize=5.9,
          color=NEUTRAL_MID, ha="right", va="bottom")
ax_b.tick_params(axis="y", length=0)
add_panel_label(ax_b, "b")

pd.DataFrame(loc_rows).to_csv(OUT / "fig2_robust_location.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(shift_rows).to_csv(OUT / "fig2_robust_shift.csv", index=False, encoding="utf-8-sig")

fig.subplots_adjust(left=0.105, right=0.985, top=0.915, bottom=0.115, wspace=0.09)
fig.canvas.draw()

require_matplotlib_panel_alignment(
    fig,
    json_out=str(OUT / "fig2_robust.alignment.json"),
    overlay_svg=str(OUT / "fig2_robust.alignment.svg"),
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
    panel_ids=["a", "b"],
    row_groups=[["a", "b"]],
)

fig.savefig(OUT / "fig2_robust.svg")
fig.savefig(OUT / "fig2_robust.pdf")
fig.savefig(OUT / "fig2_robust.png", dpi=600)
fig.savefig(OUT / "fig2_robust.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("fig2 done")



