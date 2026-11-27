"""Reproduce the descriptive tables and a box/point overview."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, FEATURES, GROUPS, SEED, load, save_figure, save_table

SECTION = "01_descriptive"


def main():
    rows = load("Data01_clean.csv")
    output = []
    by_group = rows.groupby("CallType")
    table_features = {**FEATURES, "spectral_IQR_Hz": ("Spectral interquartile bandwidth", "Hz")}
    for metric, (label, unit) in table_features.items():
        for group in GROUPS:
            all_values = by_group.get_group(group)[metric]
            values = all_values.dropna()
            stats = values.describe()
            output.append({"metric": metric, "label": label, "unit": unit, "call_type": group,
                           "n": len(values), "missing": all_values.isna().sum(),
                           "mean": stats["mean"], "sd": stats["std"], "min": stats["min"],
                           "p25": stats["25%"], "median": stats["50%"], "p75": stats["75%"], "max": stats["max"]})
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for ax, (metric, (_, unit)) in zip(axes.flat, FEATURES.items()):
        for position, group in enumerate(GROUPS, start=1):
            values = by_group.get_group(group)[metric].dropna()
            if values.empty:
                ax.text(position, 0.02, "n=0", ha="center", va="bottom", transform=ax.get_xaxis_transform(),
                        fontsize=8, color="#666666")
                continue
            ax.boxplot(values, positions=[position], widths=.5, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": COLORS[group], "alpha": .25},
                       medianprops={"color": "#222222"})
            jitter = rng.uniform(-.17, .17, len(values))
            ax.scatter(position + jitter, values, color=COLORS[group], s=8, alpha=.60, linewidths=0)
        ax.set(xlim=(.5, 3.5), xticks=[1, 2, 3], xticklabels=GROUPS, title=metric, ylabel=unit)
        ax.spines[["top", "right"]].set_visible(False)
    save_table(pd.DataFrame(output), SECTION, "Q1_descriptive_statistics.csv")
    save_figure(fig, SECTION, "descriptive_reproduced")


if __name__ == "__main__":
    main()
