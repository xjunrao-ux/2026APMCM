"""Fit and use a finite-window response latency model from Data02."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.special import log_ndtr, ndtri_exp

from _fit_q3_work import (
    BEHAVIORS, CALLS, COMBO_SD, CON, CONTEXTS, CTYPE, FIRST, INDICATORS, LAT,
    NAT, OCC_SD, PAIRS, TIME_SD, TRACK, TRACK_ID, TRACKS, W, X, Y, YCODE,
    design, fit_occ, fit_time, prob, serialize,
)
from q3_behavior_factors import PENALTY_SCALES, score_models


ENDPOINTS = (*BEHAVIORS, "Any")
FEATURES = ("intercept", "natural", "subtype_1", "subtype_2", "after", "searching",
            "log2_distance_over_80", "log2_group_size_over_8")
ZERO_UPPER = 0.5
TIME_SCALES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
MIDPOINTS = (3.0, 5.0, 10.0, 3.0)
CHECK_TIMES = np.array([3.0, 5.0, 10.0])
HELD_OUT = tuple(PAIRS[ids].ravel() for ids in TRACK_ID.values())


def time_parameters(x, fit):
    return x @ fit[:-1], np.exp(fit[-1])


def conditional_cdf(seconds, mu, sigma):
    log_window = log_ndtr((np.log(W) - mu) / sigma)
    seconds = np.asarray(seconds, dtype=float)
    interior = np.clip(seconds, np.finfo(float).tiny, W)
    cdf = np.exp(log_ndtr((np.log(interior) - mu) / sigma) - log_window)
    return np.where(seconds <= 0, 0.0, cdf)


def conditional_median(mu, sigma):
    log_window = log_ndtr((np.log(W) - mu) / sigma)
    return np.exp(mu + sigma * ndtri_exp(np.log(0.5) + log_window))


def conditional_loglik(time, mu, sigma, zero_upper):
    zero = time == 0
    log_time = np.log(np.where(zero, zero_upper, time))
    z = (log_time - mu) / sigma
    log_density = -0.5 * z**2 - np.log(sigma) - log_time - 0.5 * np.log(2 * np.pi)
    return np.where(zero, log_ndtr(z), log_density) - log_ndtr((np.log(W) - mu) / sigma)


def fit_endpoint(columns, time, weight, midpoint, scale, zero_upper=ZERO_UPPER, start=None):
    x = X[:, columns]
    included = np.isfinite(time) & (weight > 0)
    sd = TIME_SD[columns].copy()
    sd[1:] *= scale
    return fit_time(x[included], time[included], weight[included], midpoint, zero_upper,
                    start=start, coefficient_sd=sd)


def validate_time(columns, time, midpoint, candidates):
    cdf = np.empty((len(candidates), len(time), len(CHECK_TIMES)))
    loss = np.empty((len(candidates), len(time)))
    for held in HELD_OUT:
        rows = held[np.isfinite(time[held])]
        if not len(rows):
            continue
        weights = np.ones(len(time))
        weights[held] = 0
        for i, scale in enumerate(candidates):
            terms = columns if scale > 0 else columns[:1]
            fit = fit_endpoint(terms, time, weights, midpoint, scale)
            mu, sigma = time_parameters(X[np.ix_(rows, terms)], fit)
            cdf[i, rows] = conditional_cdf(CHECK_TIMES, mu[:, None], sigma)
            loss[i, rows] = -conditional_loglik(time[rows], mu, sigma, ZERO_UPPER)
    valid = np.isfinite(time)
    score = loss[:, valid].mean(axis=1)
    best = np.argmin(score)
    return candidates[best], {
        "conditional_log_loss": score,
        "conditional_cdf_checks": {
            "seconds": CHECK_TIMES,
            "held_out_prediction": cdf[best, valid].mean(axis=0),
            "observed": (time[valid, None] <= CHECK_TIMES).mean(axis=0),
        },
    }


def track_bootstrap(repeats):
    rng = np.random.default_rng(7112026)
    for _ in range(repeats):
        tracks = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in TRACKS])
        unique, counts = np.unique(tracks, return_counts=True)
        by_track = dict(zip(unique, counts))
        per_pair = [by_track.get(track, 0) for track in TRACK]
        w = np.zeros(len(YCODE))
        w[NAT] = w[CON] = per_pair
        yield w


def display_parameters(coefficients, columns, reference):
    # The report uses sound differences and uncentered log2(group size).
    beta = np.zeros((*coefficients.shape[:-1], len(FEATURES)))
    beta[..., columns] = coefficients[..., :-1]
    sound_offset = beta @ design(np.arange(4), 0, 80.0, 8.0)[:, :8].T - beta[..., :1]
    base = beta[..., 0] + sound_offset[..., reference] - 3 * beta[..., 7]
    sounds = [c for c in range(4) if c != reference and (reference == 0 or c > 0)]
    differences = [sound_offset[..., c] - sound_offset[..., reference] for c in sounds]
    names = ["intercept", *(f"{CALLS[c]}_vs_{CALLS[reference]}" for c in sounds),
             "after_vs_commuting", "searching_vs_commuting",
             "log2_distance_over_80", "log2_group_size", "sigma"]
    values = np.stack([base, *differences, *np.moveaxis(beta[..., 4:8], -1, 0),
                       np.exp(coefficients[..., -1])], axis=-1)
    return names, values


def fit_model(repeats):
    scores = score_models(X[:, :8], OCC_SD[:8], COMBO_SD[:8])
    occ_scale, combo_scale = np.asarray(PENALTY_SCALES)[scores.argmin(axis=0)]
    weights = np.ones(len(YCODE))
    binary, combo = fit_occ(weights, X[:, :8], OCC_SD[:8] * occ_scale,
                            COMBO_SD[:8] * combo_scale)
    observed = np.column_stack([np.where(Y == 1, LAT, np.nan),
                                np.where(np.isfinite(FIRST), FIRST, np.nan)])
    time_fits = {}
    times = {}
    for k, (key, midpoint) in enumerate(zip(ENDPOINTS, MIDPOINTS)):
        reference = int(key == "Approach")
        columns = np.array([0, 2, 3, 4, 5, 6, 7]) if reference else np.arange(8)
        time = observed[:, k]
        scale, validation = validate_time(columns, time, midpoint, (0.0, *TIME_SCALES))
        if scale == 0:
            columns = columns[:1]
        coefficients = fit_endpoint(columns, time, weights, midpoint, scale)
        time_fits[key] = (columns, time, midpoint, scale, coefficients)
        records = [time[CTYPE == call] for call in range(len(CALLS))]
        times[key] = {
            "features": columns,
            "coefficients": coefficients,
            "reference": CALLS[reference],
            "slope_sd_scale": scale,
            "validation": validation,
            "events_by_sound": [np.isfinite(records_for_sound).sum() for records_for_sound in records],
            "observed_median_by_sound": [
                np.nanmedian(records_for_sound) if np.isfinite(records_for_sound).any() else None
                for records_for_sound in records
            ],
            "zero_events": np.count_nonzero(time == 0),
        }
        print(f"{key}: {np.isfinite(time).sum()} events, slope SD scale {scale:g}", flush=True)
    draws = {key: [] for key in ENDPOINTS}
    for w in track_bootstrap(repeats):
        for key, (columns, time, midpoint, scale, coefficients) in time_fits.items():
            draws[key].append(fit_endpoint(columns, time, w, midpoint, scale, start=coefficients))
    for key, (columns, time, midpoint, scale, coefficients) in time_fits.items():
        reference = CALLS.index(times[key]["reference"])
        names, values = display_parameters(coefficients, columns, reference)
        _, boot = display_parameters(np.asarray(draws[key]), columns, reference)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        times[key]["report_coefficients"] = dict(zip(names, np.stack([values, lower, upper], axis=-1)))
        times[key]["zero_window_sensitivity"] = {}
        for upper_zero in (0.25, 1.0) if (time == 0).any() else ():
            # Keep the first positive record outside the assumed zero interval.
            if np.min(time[time > 0]) < upper_zero:
                continue
            alternative = fit_endpoint(columns, time, weights, midpoint, scale, zero_upper=upper_zero)
            _names, delta = display_parameters(alternative, columns, reference)
            times[key]["zero_window_sensitivity"][str(upper_zero)] = dict(zip(names, delta - values))
    return {
        "observation_window_seconds": W,
        "recorded_zero_interval_upper_seconds": ZERO_UPPER,
        "feature_names": FEATURES,
        "sound_names": CALLS,
        "context_names": CONTEXTS,
        "positive_combinations": ["A", "V", "V+A", "H", "H+A", "H+V", "H+V+A"],
        "repeats": repeats,
        "scipy_version": scipy.__version__,
        "occurrence": {
            "binary_coefficients": binary,
            "combination_coefficients": combo,
            "scales": [occ_scale, combo_scale],
            "leave_track_out_log_loss": scores.min(axis=0),
        },
        "time_slope_scale_candidates": [0.0, *TIME_SCALES],
        "time": times,
    }


def predict(model, sound, context, distance, group_size, seconds):
    """Predict at caller-supplied conditions and times inside the observation window."""
    seconds = np.asarray(seconds, dtype=float)
    if not (np.isfinite(distance) and np.isfinite(group_size) and distance > 0 and group_size > 0):
        raise ValueError("Distance and group size must be positive and finite.")
    if not np.all(np.isfinite(seconds) & (seconds >= 0) & (seconds <= W)):
        raise ValueError(f"Prediction times must be between 0 and {W:g} seconds.")
    x = design(np.array([CALLS.index(sound)]), CONTEXTS.index(context), distance, group_size)[0, :8]
    occ = model["occurrence"]
    combo = prob(x, (np.asarray(occ["binary_coefficients"]), np.asarray(occ["combination_coefficients"])))
    p = np.r_[combo @ INDICATORS, 1 - combo[0]]
    result = {}
    for i, key in enumerate(ENDPOINTS):
        fit = model["time"][key]
        observed = fit["events_by_sound"][CALLS.index(sound)]
        info = {"probability_by_60": p[i], "observed_events_for_sound": observed,
                "conditional_median_seconds": None, "cumulative_probability": None}
        if observed:
            mu, sigma = time_parameters(x[np.asarray(fit["features"])], np.asarray(fit["coefficients"]))
            info["conditional_median_seconds"] = conditional_median(mu, sigma)
            info["cumulative_probability"] = p[i] * conditional_cdf(seconds, mu, sigma)
        result[key] = info
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    model = fit_model(args.bootstrap)
    args.output.write_text(json.dumps(model, default=serialize, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
