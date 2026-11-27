"""Shared paths and output conventions for the Question 1 analyses."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GROUPS = ("flat", "mod", "tv")
COLORS = {"flat": "#0072B2", "mod": "#D55E00", "tv": "#009E73", "control": "#777777"}
COMMON = ("Duration", "F01", "Fpeak", "Q25", "Q50", "Q75")
FEATURES = {"Duration": ("Call duration", "s"), "F01": ("First fundamental frequency", "Hz"),
            "F02": ("Second fundamental frequency", "Hz"), "Fpeak": ("Peak frequency", "Hz"),
            "Q25": ("25% energy frequency", "Hz"), "Q50": ("50% energy frequency", "Hz"),
            "Q75": ("75% energy frequency", "Hz"), "PulseRate": ("Pulse rate", "pulses/s")}
SEED = 20261126


def load(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def save_table(data, section, name):
    data.to_csv(ROOT / section / "tables" / name, index=False, encoding="utf-8-sig")


def save_figure(fig, section, name):
    path = ROOT / section / "figures" / name
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    index = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[index] = np.minimum(1, np.maximum.accumulate(values[index] * np.arange(len(values), 0, -1)))
    return adjusted
