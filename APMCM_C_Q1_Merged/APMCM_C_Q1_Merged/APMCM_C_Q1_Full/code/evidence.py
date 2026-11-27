"""Reproduce the propagation and playback summaries used in Question 1."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, GROUPS, load, save_figure, save_table

SECTION = "04_function_and_transmission"
BEHAVIORS = {"ResponseYesNo": "Any response", "HeadsUpYesNo": "Heads up",
             "VocalYesNo": "Vocal response", "ApproachYesNo": "Approach"}
MEASURES = {"spectrogram": "Spectrogram", "amplitude_spectrum": "Amplitude spectrum",
            "amplitude_envelope": "Amplitude envelope"}


def playback_summary(rows):
    table = []
    for group in (*GROUPS, "control"):
        samples = rows[rows.CallType.eq(group)]
        for outcome, title in BEHAVIORS.items():
            table.append({"call_type": group, "outcome": outcome, "label": title, "n": len(samples),
                          "positive": int(samples[outcome].sum()), "proportion": samples[outcome].mean()})
    return pd.DataFrame(table)


def playback_fig(table):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True, layout="constrained")
    groups = (*GROUPS, "control")
    for ax, (outcome, label) in zip(axes, list(BEHAVIORS.items())[1:]):
        t = table.loc[table.outcome.eq(outcome)].set_index("call_type").loc[list(groups)]
        bars = ax.bar(groups, t.proportion * 100, color=[COLORS[g] for g in groups], width=.65)
        ax.bar_label(bars, [f"{v}/{n}" for v, n in zip(t.positive, t.n)], fontsize=8, padding=4)
        ax.set(title=label, ylim=(0, 72))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Positive playbacks (%)")
    save_figure(fig, SECTION, "playback_response")


def propagation_summary(rows):
    grouped = rows.groupby(["measure", "call_type", "distance_m"]).correlation
    return grouped.agg(total="size", valid="count", mean="mean", sd="std", median="median").reset_index()


def propagation_fig(table):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    for ax, (measure, label) in zip(axes, MEASURES.items()):
        for group in GROUPS:
            t = table.loc[table.measure.eq(measure) & table.call_type.eq(group)].sort_values("distance_m")
            ax.plot(t.distance_m, t["mean"], marker="o", markersize=3.5, label=group, color=COLORS[group])
        ax.set_xscale("log", base=2)
        if measure == "spectrogram":
            ax.axhline(.5, ls="--", lw=.8, color=".45")
        ax.set(xticks=[4, 8, 16, 32, 64, 128, 256], xticklabels=[4, 8, 16, 32, 64, 128, 256],
               ylim=(0, 1), title=label, xlabel="Distance (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean correlation to the 1 m reference")
    axes[-1].legend(frameon=False, loc="lower left")
    save_figure(fig, SECTION, "propagation_profiles")


def main():
    rows = load("Data02_clean.csv")
    calls = rows.loc[rows.Type.eq("call")].set_index("expID")
    control = rows.loc[rows.Type.eq("control")].set_index("expID").loc[calls.index]
    assert len(calls) == len(control) == 139 and calls.index.is_unique and control.index.is_unique
    paired = []
    for group in GROUPS:
        call = calls.loc[calls.CallType.eq(group)]
        baseline = control.loc[call.index]
        for outcome in BEHAVIORS:
            paired.append({"type": group, "outcome": outcome, "pairs": len(call),
                           "call_positive": int(call[outcome].sum()), "control_positive": int(baseline[outcome].sum()),
                           "call_only": int((call[outcome].eq(1) & baseline[outcome].eq(0)).sum()),
                           "control_only": int((call[outcome].eq(0) & baseline[outcome].eq(1)).sum())})
    behavior = playback_summary(rows)
    save_table(behavior, SECTION, "playback_by_stimulus.csv")
    save_table(pd.DataFrame(paired), SECTION, "matched_playback_counts.csv")
    by_context = calls.groupby("Context")[list(BEHAVIORS)]
    save_table(by_context.sum().assign(n=by_context.size()).reset_index(),
               SECTION, "natural_playback_by_context.csv")
    playback_fig(behavior)
    propagation = propagation_summary(load("Data03-11_propagation_long.csv"))
    thresholds = []
    for group in GROUPS:
        spec = propagation.loc[propagation.measure.eq("spectrogram") & propagation.call_type.eq(group)]
        spec = spec.sort_values("distance_m").reset_index(drop=True)
        first_below = int(np.flatnonzero(spec["mean"].to_numpy() < .5)[0])
        thresholds.append({"call_type": group, "threshold": .5,
                           "lower_distance_m": spec.loc[first_below - 1, "distance_m"],
                           "upper_distance_m": spec.loc[first_below, "distance_m"]})
    save_table(propagation, SECTION, "propagation_by_measure_distance.csv")
    save_table(pd.DataFrame(thresholds), SECTION, "spectrogram_0p5_brackets.csv")
    propagation_fig(propagation)


if __name__ == "__main__":
    main()
