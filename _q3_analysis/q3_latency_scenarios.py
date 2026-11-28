"""Predict first-response times for the representative playback conditions."""

import argparse
import json
from pathlib import Path

import numpy as np

from _fit_q3_work import (
    CALLS, COMBO_SD, CONTEXTS, CONTEXT_WEIGHT, CTYPE, D, FIRST, INDICATORS,
    LAT, N, OCC_SD, W, X, Y, YCODE, design, fit_occ, prob, serialize,
)
from q3_latency_model import (
    ENDPOINTS, MIDPOINTS, conditional_cdf, fit_endpoint, time_parameters, track_bootstrap,
)


DISTANCES = (50, 100, 150)
CONDITIONS = tuple((c, d) for c in range(len(CALLS)) for d in DISTANCES)
EVENTS = np.column_stack([Y, YCODE > 0])
TIMES = np.column_stack([np.where(Y, LAT, np.nan), np.where(np.isfinite(FIRST), FIRST, np.nan)])


def scenario_design(group_size):
    return np.stack([
        design(c, np.arange(len(CONTEXTS)), distance, group_size)[:, :8]
        for c, distance in CONDITIONS
    ])


def scenario_predictions(x, occurrence, time_models):
    combos = prob(x, occurrence)
    p = np.concatenate([combos @ INDICATORS, (1 - combos[..., :1])], axis=-1)
    context_mass = p * CONTEXT_WEIGHT[None, :, None]
    window_probability = context_mass.sum(axis=1)
    parameters = [time_parameters(x[..., terms], coef) for terms, coef in time_models]
    mu = np.stack([param[0] for param in parameters], axis=-1)
    sigma = np.array([param[1] for param in parameters])
    lo = np.zeros_like(window_probability)
    hi = np.full_like(window_probability, W)
    for _ in range(26):
        t = (lo + hi) / 2
        cumulative = (context_mass * conditional_cdf(t[:, None, :], mu, sigma)).sum(axis=1)
        before_median = cumulative < window_probability / 2
        lo = np.where(before_median, t, lo)
        hi = np.where(before_median, hi, t)
    median = (lo + hi) / 2
    check = (context_mass * conditional_cdf(median[:, None, :], mu, sigma)).sum(axis=1)
    np.testing.assert_allclose(check, window_probability / 2, atol=3e-7)
    return window_probability, median


def observed_support(call, distance, group_size, endpoint):
    at_distance = (CTYPE == call) & (D == distance)
    exact = at_distance & (N == group_size)
    raw = {
        "distance_sample": at_distance.sum(),
        "distance_events": EVENTS[at_distance, endpoint].sum(),
        "exact_sample": exact.sum(),
        "exact_events": EVENTS[exact, endpoint].sum(),
    }
    if raw["exact_sample"]:
        status = "exact_observed" if raw["exact_events"] else "no_response_exact"
    elif raw["distance_sample"] and not raw["distance_events"]:
        status = "no_response_at_distance"
    else:
        status = "prediction_only"
    return raw, status


def summarize(point, samples):
    return np.stack([point, *np.quantile(samples, (0.025, 0.975), axis=0)], axis=-1)


def calculate(model):
    size = float(np.median(N))
    x = scenario_design(size)
    occ = model["occurrence"]
    occurrence = tuple(np.asarray(occ[key]) for key in ("binary_coefficients", "combination_coefficients"))
    occ_scale, combo_scale = occ["scales"]
    time_models = [
        (np.asarray(model["time"][endpoint]["features"]), np.asarray(model["time"][endpoint]["coefficients"]))
        for endpoint in ENDPOINTS
    ]
    time_scales = [model["time"][endpoint]["slope_sd_scale"] for endpoint in ENDPOINTS]
    zero_upper = model["recorded_zero_interval_upper_seconds"]
    p, median = scenario_predictions(x, occurrence, time_models)
    p_draws, median_draws = [], []
    for w in track_bootstrap(model["repeats"]):
        use = w > 0
        fitted_occ = fit_occ(w[use], X[use, :8], OCC_SD[:8] * occ_scale,
                             COMBO_SD[:8] * combo_scale, codes=YCODE[use], start=occurrence)
        fits = [
            (terms, fit_endpoint(terms, TIMES[:, i], w, MIDPOINTS[i], time_scales[i],
                                 zero_upper, start=coef))
            for i, (terms, coef) in enumerate(time_models)
        ]
        sample_p, sample_median = scenario_predictions(x, fitted_occ, fits)
        p_draws.append(sample_p)
        median_draws.append(sample_median)
    p_table = summarize(p, p_draws)
    median_table = summarize(median, median_draws)
    rows = []
    for i, (call, distance) in enumerate(CONDITIONS):
        outcomes = {}
        for j, endpoint in enumerate(ENDPOINTS):
            support, status = observed_support(call, distance, size, j)
            display_time = status in ("exact_observed", "prediction_only")
            outcomes[endpoint] = {
                **support,
                "status": status,
                "probability_by_60": p_table[i, j],
                "reported_conditional_median_seconds": median_table[i, j] if display_time else None,
            }
        rows.append({"sound": CALLS[call], "distance_m": distance, "outcomes": outcomes})
    # Compare the two extremes within each resample, retaining the joint uncertainty.
    time_any = median[:, -1].reshape(len(CALLS), len(DISTANCES))
    draw_any = np.asarray(median_draws)[..., -1].reshape(-1, len(CALLS), len(DISTANCES))
    time_change = summarize(time_any[:, -1] - time_any[:, 0], draw_any[..., -1] - draw_any[..., 0])
    return {
        "group_size": size,
        "observation_window_seconds": W,
        "context_weights": dict(zip(CONTEXTS, CONTEXT_WEIGHT)),
        "repeats": model["repeats"],
        "interval": "95% pointwise track bootstrap percentile",
        "rows": rows,
        "any_latency_change_50_to_150_seconds": dict(zip(CALLS, time_change)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("q3_latency_model_results.json"))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    results = calculate(model)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=serialize,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(results['rows'])} conditions at Npenguins={results['group_size']:g}, "
          f"{results['repeats']} track bootstrap draws", flush=True)


if __name__ == "__main__":
    main()
