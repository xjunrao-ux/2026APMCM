"""Shared paths and model inputs for the three playback figures."""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from plot_q3_summary import COMBINATION_LABELS, COMBINATION_ORDER, read_results

OUT = ANALYSIS.parent / "figures" / "q3_revised"


def inputs():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 8,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    return read_results(ANALYSIS / "behavior_factor_results.json")
