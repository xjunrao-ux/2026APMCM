"""Standardize single-behavior probabilities from the two-stage playback model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _fit_q3_work import (
    CALLS,
    COMBO_SD,
    CON,
    CTX,
    CTYPE,
    D,
    DATA,
    INDICATORS,
    N,
    NAT,
    OCC_SD,
    PAIRS,
    ROOT,
    TRACK,
    TRACK_ID,
    TRACKS,
    Y,
    YCODE,
    design,
    fit_occ,
    prob,
    serialize,
    simultaneous,
)


CONTEXT_COMPARISONS = ((1, 0), (2, 0))
SCENARIOS = {
    "observed": ({},),
    "context": tuple({"context": b} for b in range(3)),
    "distance": ({"distance": 50}, {"distance": 100}),
    "size": ({"size": 4}, {"size": 8}),
}
CONTEXT_SUPPORT = (N >= 2) & (N <= 10) & (D >= 20) & (D <= 150)
PAIRS_OF_SOUNDS = ((2, 1), (3, 1), (3, 2))
PENALTY_SCALES = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0)


def features(call, context, distance, size, allow_interactions):
    x = design(call, context, distance, size)
    if allow_interactions:
        return np.column_stack([x, x[:, 1] * x[:, 7]])
    return x[:, :8]


def build_scenarios(allow_interactions):
    fields = {"context": CTX, "distance": D, "size": N}
    scenarios = {}
    for key, settings in SCENARIOS.items():
        rows = {name: values[CONTEXT_SUPPORT] for name, values in fields.items()} if key == "context" else fields
        scenarios[key] = np.stack([
            np.stack([features(c, **(rows | overrides), allow_interactions=allow_interactions)
                      for c in range(len(CALLS))])
            for overrides in settings
        ])
    return scenarios


def probabilities(model, scenarios, validate=False):
    out = {}
    combined = {}
    for key, x in scenarios.items():
        combo = prob(x, model).mean(axis=2)
        if validate:
            np.testing.assert_allclose(combo.sum(axis=-1), 1, atol=1e-12)
        combined[key] = combo
        behavior = combo @ INDICATORS
        out[key] = np.concatenate([behavior, behavior[:, 1:].mean(axis=1, keepdims=True)], axis=1)
    return out, combined


def contrasts(p):
    all_sounds = p["observed"][0]
    context = p["context"]
    deltas = {
        "sound": all_sounds[1:] - all_sounds[0],
        "natural_pairs": np.array([all_sounds[a] - all_sounds[b] for a, b in PAIRS_OF_SOUNDS]),
        "context": np.array([context[a] - context[b] for a, b in CONTEXT_COMPARISONS]),
        "distance": p["distance"][1:] - p["distance"][:1],
        "size": p["size"][1:] - p["size"][:1],
    }
    for factor in ("context", "distance", "size"):
        d = deltas[factor]
        control, natural = d[:, 0], d[:, 4]
        deltas[factor] = np.stack([control, natural, natural - control], axis=1)
    return deltas


def score_models(x, occurrence_sd, combo_sd):
    """Leave out one natural recording and all its experiment pairs in each fold."""
    losses = np.zeros((len(PENALTY_SCALES), 2))
    for ids in TRACK_ID.values():
        held = PAIRS[ids].ravel()
        weights = np.ones(len(YCODE))
        weights[held] = 0
        active = YCODE[held] > 0
        for i, scale in enumerate(PENALTY_SCALES):
            model = fit_occ(weights, x, occurrence_sd * scale, combo_sd * scale)
            p = prob(x[held], model)
            q = p[np.arange(len(held)), YCODE[held]]
            p_active = 1 - p[active, 0]
            losses[i, 0] -= np.log(q[~active]).sum() + np.log(p_active).sum()
            losses[i, 1] -= (np.log(q[active]) - np.log(p_active)).sum()
    return losses / len(YCODE)


def fit_settings(n_boot, seed, interactions):
    x = features(CTYPE, CTX, D, N, interactions)
    scenarios = build_scenarios(interactions)
    occurrence_sd = np.r_[OCC_SD, .65] if interactions else OCC_SD[:8]
    combo_sd = np.r_[COMBO_SD, .55] if interactions else COMBO_SD[:8]
    scores = score_models(x, occurrence_sd, combo_sd)
    scale_occ, scale_combo = np.array(PENALTY_SCALES)[scores.argmin(axis=0)]
    occurrence_sd = occurrence_sd * scale_occ
    combo_sd = combo_sd * scale_combo
    fit = fit_occ(np.ones(len(YCODE)), x, occurrence_sd, combo_sd)
    p, combo = probabilities(fit, scenarios, validate=True)
    point = contrasts(p)
    print(f"penalty scales {scale_occ}, {scale_combo}; resampling {n_boot} track sets", flush=True)
    draws = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in TRACKS])
        unique, counts = np.unique(sampled, return_counts=True)
        per_track = dict(zip(unique, counts))
        weights = np.zeros(len(YCODE))
        per_pair = [per_track.get(t, 0) for t in TRACK]
        weights[NAT] = weights[CON] = per_pair
        selected = weights > 0
        boot_fit = fit_occ(weights[selected], x[selected], occurrence_sd, combo_sd,
                           codes=YCODE[selected], start=fit)
        boot_point = contrasts(probabilities(boot_fit, scenarios)[0])
        for key, value in boot_point.items():
            draws[key].append(value)
    return {
        "penalty_scales": [scale_occ, scale_combo],
        "scores": scores,
        "cv_log_loss": scores.min(axis=0).sum(),
        "behavior_probabilities": p,
        "combination_probabilities": combo["observed"][0],
        "contrasts": {key: simultaneous(value, draws[key]) for key, value in point.items()},
    }


def published_probabilities():
    file = ROOT / "_q3_analysis/code1_results/Q3_2_Code1_All_Contrasts.csv"
    codes = {"No response": 0, "A": 1, "V": 2, "V+A": 3, "H": 4,
             "H+A": 5, "H+V": 6, "H+V+A": 7}
    combo = np.zeros((5, 8))
    names = (*CALLS, "natural_equal")
    with file.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["reference"] != "control":
                continue
            code = codes[row["combination"]]
            combo[0, code] = float(row["reference_probability_pct"]) / 100
            combo[names.index(row["stimulus"]), code] = float(row["stimulus_probability_pct"]) / 100
    np.testing.assert_allclose(combo.sum(axis=1), 1, atol=1e-8)
    return combo @ INDICATORS


def observed():
    result = {}
    for name, groups in (
        ("sound", DATA.CallType),
        ("context", DATA.Context),
        ("distance", np.select([D <= 50, D <= 80, D <= 100], ["1–50", "51–80", "81–100"], ">100")),
        ("size", np.select([N <= 4, N <= 9, N <= 19], ["1–4", "5–9", "10–19"], "≥20")),
    ):
        result[name] = {}
        for key in np.unique(groups):
            result[name][key] = {}
            for label in ("call", "control"):
                mask = (groups == key) & (DATA.Type == label).to_numpy()
                if mask.any():
                    result[name][key][label] = [mask.sum(), *Y[mask].sum(axis=0)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    seed = 7112026
    report = {
        "bootstrap": args.bootstrap,
        "observed": observed(),
        "published": published_probabilities(),
        "penalty_grid": PENALTY_SCALES,
        "context_standardization_n": CONTEXT_SUPPORT.sum(),
    }
    for name, interactions in (("additive", False), ("interactions", True)):
        report[name] = fit_settings(args.bootstrap, seed, interactions)
        print(f"{name}: grouped CV log loss {report[name]['cv_log_loss']:.5f}", flush=True)
    Path(args.output).write_text(json.dumps(report, default=serialize, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
