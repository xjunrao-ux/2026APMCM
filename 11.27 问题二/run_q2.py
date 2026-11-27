"""Run the approved propagation analysis. Python 3.11+, numpy, scipy, pandas,
matplotlib; R with mgcv. Use --rscript if Rscript is not on PATH.
"""
import argparse
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from scipy.stats import t

from 核验第二问方案 import CALLS, DATA, DISTANCE, fit, load_correlations, predict
from q2_figures import draw_figures


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
METRICS = ["spectrogram", "spectrum", "envelope"]
LEVELS = (0.5, 0.6, 0.7)
GRID = np.linspace(4, 256, 2521)


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig",
                 float_format="%.10g")


def audit():
    values = load_correlations()
    finite = np.isfinite(values)
    assert np.all((values[finite] >= -1) & (values[finite] <= 1))
    records = []
    for ci, call in enumerate(CALLS):
        for mi, metric in enumerate(METRICS):
            number = 3 + ci + 3 * mi
            frame = pd.DataFrame(values[ci, mi], columns=DISTANCE.astype(int))
            frame.index = np.arange(1, len(frame) + 1)
            frame = frame.rename_axis("source_row").reset_index().melt(
                id_vars="source_row", var_name="d", value_name="correlation")
            records.append(frame.assign(call=call, metric=metric, source=f"Data{number:02d}"))
    raw = pd.concat(records, ignore_index=True)
    raw["d"] = raw.d.astype(int)
    save(raw, "propagation_long")
    stats = raw.groupby(["call", "metric", "d"]).correlation.agg(
        slots="size", valid="count", mean="mean", median="median", sd="std",
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
        minimum="min", maximum="max").reset_index()
    stats["missing"] = stats.slots - stats.valid
    stats["missing_fraction"] = stats.missing / stats.slots
    save(stats, "audit_63")
    return values


def pooled(means):
    return np.apply_along_axis(
        lambda row: isotonic_regression(row, increasing=False).x, -1, means)


def interpolate(values, x, q, log=True):
    if np.min(q) < x[0] or np.max(q) > x[-1]:
        raise ValueError("Outside measured support")
    source, query = (np.log2(x), np.log2(q)) if log else (x, q)
    return np.apply_along_axis(lambda row: np.interp(query, source, row), -1, values)


def fidelity(curves, w=1 / 3):
    return (1 - w) * curves[:, :2].mean(axis=1) + w * curves[:, 2]


def crossing(x, y, threshold, log=True):
    """First loss of continuous coverage; also works with nonmonotone GAM curves."""
    below = np.flatnonzero(y < threshold)
    if not len(below):
        return dict(distance_m=x[-1], status="right_censored", left_m=x[-1], right_m=np.nan)
    j = below[0]
    if j == 0:
        return dict(distance_m=x[0], status="left_censored", left_m=np.nan, right_m=x[0])
    f = (threshold - y[j-1]) / (y[j] - y[j-1])
    left, right = x[j-1:j+1]
    d = left * (right / left)**f if log else left + f * (right - left)
    return dict(distance_m=d, status="crossing", left_m=left, right_m=right)


def threshold_analysis(means, fits):
    raw_f, fit_f = fidelity(means), fidelity(fits)
    ranges, sweeps, folds, weights, reversals = [], [], [], [], []
    for ci, call in enumerate(CALLS):
        for level in np.arange(0, 1001) / 1000:
            point = crossing(DISTANCE, fit_f[ci], level)
            sweeps.append(dict(call=call, threshold=level, **point))
        for level in LEVELS:
            bad = np.flatnonzero(raw_f[ci] < level)
            n_pass = int(bad[0]) if bad.size else len(DISTANCE)
            point = crossing(DISTANCE, fit_f[ci], level)
            ranges.append(dict(call=call, threshold=level, **point,
                linear_m=crossing(DISTANCE, fit_f[ci], level, False)["distance_m"],
                last_consecutive_observed_m=DISTANCE[n_pass-1] if n_pass else np.nan,
                first_failed_observed_m=DISTANCE[n_pass] if n_pass<len(DISTANCE) else np.nan))
        for w in np.unique(np.r_[np.linspace(0, 1, 101), 1/3]):
            for level in LEVELS:
                weights.append(dict(call=call, threshold=level, envelope_weight=w,
                    **crossing(DISTANCE, fidelity(fits, w)[ci], level)))
    for drop, distance in enumerate(DISTANCE):
        keep = np.arange(7) != drop
        reduced = fidelity(pooled(means[:, :, keep]))
        for ci, call in enumerate(CALLS):
            for level in LEVELS:
                folds.append(dict(call=call, threshold=level, omitted_m=distance,
                    **crossing(DISTANCE[keep], reduced[ci], level)))
    for basis, curves in [("observed", means), ("monotone", fits)]:
        for a, b in itertools.combinations(range(3), 2):
            spec = curves[:, :2].mean(axis=1)
            for i, d in enumerate(DISTANCE):
                ds, de = spec[a,i]-spec[b,i], curves[a,2,i]-curves[b,2,i]
                switch = ds / (ds-de) if ds * de < 0 else np.nan
                reversals.append(dict(basis=basis, d=d, call_a=CALLS[a], call_b=CALLS[b],
                    spectral_advantage_a=ds, envelope_advantage_a=de, flip_weight=switch))
    ranges, folds = pd.DataFrame(ranges), pd.DataFrame(folds)
    exact = folds[folds.status == "crossing"].groupby(["call","threshold"])
    limits = exact.distance_m.agg(delete_min_m="min", delete_max_m="max", n_exact="count")
    ranges = ranges.merge(limits, on=["call","threshold"], how="left")
    for frame, name in [(ranges,"threshold_ranges"), (sweeps,"threshold_sweep"),
                        (folds,"delete_distance"), (weights,"weight_sensitivity"),
                        (reversals,"weight_reversals")]:
        save(pd.DataFrame(frame), name)
    return ranges, pd.DataFrame(sweeps)


def compare_models(means, gam):
    folds, full_ranges = [], []
    for omitted in [0, 8, 16, 32, 64, 128]:
        keep = DISTANCE != omitted
        x = GRID if omitted == 0 else np.array([float(omitted)])
        iso = pooled(means[:, :, keep])
        # Geometric spreading + stationary noise, fitted near-field ceilings.
        # Spectrogram and spectrum share the effective length; envelope has its own.
        physics = fit(means, keep, ([0,1],[2]), [0.0])
        if omitted == 0:
            parameters = [dict(call=CALLS[ci],metric=METRICS[mi],
                               length_m=length,high_snr_ceiling=ceiling)
                          for ci, metrics, length, ceilings in physics[1]
                          for mi, ceiling in zip(metrics,ceilings)]
            save(pd.DataFrame(parameters),"snr_parameters")
        gm = gam.loc[gam.hold == omitted]
        gam_curve = np.array([[gm.loc[(gm.call==call) & (gm.metric==metric)]
                               .sort_values("d").estimate.to_numpy()
                               for metric in METRICS] for call in CALLS])
        curves = {"isotonic_log": interpolate(iso, DISTANCE[keep], x),
                  "isotonic_linear": interpolate(iso, DISTANCE[keep], x, log=False),
                  "snr": predict(physics, x), "gam_mgcv": gam_curve}
        for model, predictions in curves.items():
            assert predictions.shape == (3, 3, len(x))
            if omitted:
                observed = means[:,:,np.flatnonzero(DISTANCE==omitted)[0]]
                for ci, call in enumerate(CALLS):
                    for mi, metric in enumerate(METRICS):
                        folds.append(dict(model=model, omitted_m=omitted, call=call,
                            metric=metric, observed=observed[ci,mi],
                            predicted=predictions[ci,mi,0]))
            else:
                for ci, call in enumerate(CALLS):
                    for level in LEVELS:
                        full_ranges.append(dict(model=model, call=call, threshold=level,
                            **crossing(x, fidelity(predictions)[ci], level, log=False)))
    detail = pd.DataFrame(folds)
    detail["error"] = detail.predicted - detail.observed
    save(detail, "cv_predictions")
    errors = detail.assign(sq=detail.error**2, ab=detail.error.abs())
    scores = []
    for keys, grouped in [(["model"], errors.assign(metric="all",call="all")),
                          (["model","metric"], errors.assign(call="all")),
                          (["model","call"], errors.assign(metric="all"))]:
        for _, data in grouped.groupby(keys):
            scores.append(dict(model=data.model.iloc[0],metric=data.metric.iloc[0],
                call=data.call.iloc[0],n=len(data),rmse=np.sqrt(data.sq.mean()),mae=data.ab.mean()))
    save(pd.DataFrame(scores),"cv_scores")
    save(pd.DataFrame(full_ranges),"model_thresholds")


def check_missingness(values):
    mean = np.nanmean(values, axis=2)
    fraction = np.isnan(values).mean(axis=2)
    scenarios = {"available":mean,
                 "missing_at_observed_q25":mean*(1-fraction)+fraction*np.nanquantile(values,.25,axis=2),
                 "missing_at_zero":mean*(1-fraction)}
    rows = []
    for scenario, cell_means in scenarios.items():
        trends = fidelity(pooled(cell_means))
        for ci, call in enumerate(CALLS):
            for level in LEVELS:
                rows.append(dict(scenario=scenario,call=call,threshold=level,
                                 **crossing(DISTANCE,trends[ci],level)))
    save(pd.DataFrame(rows),"missingness_sensitivity")


def playback(fits):
    raw = pd.read_csv(next(DATA.glob("Data02_*.csv")))
    key = ["Date","expID"]
    calls = raw[raw.Type=="call"].set_index(key).sort_index()
    controls = raw[raw.Type=="control"].set_index(key).sort_index()
    assert calls.index.is_unique and controls.index.is_unique and calls.index.equals(controls.index)
    assert calls.Context.equals(controls.Context)
    pairs = calls[["CallType","Distance","Context","Npenguins"]].copy()
    pairs["control_distance"] = controls.Distance
    pairs["control_group_size"] = controls.Npenguins
    pairs["fidelity"] = [float(interpolate(fidelity(fits)[CALLS.index(c)], DISTANCE, d))
                         for c,d in zip(calls.CallType, calls.Distance)]
    pairs["state"] = pd.cut(pairs.fidelity, [-np.inf,.5,.6,.7,np.inf],
        labels=["below_0.5","potential","ordinary","reliable"], right=False)
    summary, adjusted = [], []
    for field in ["ResponseYesNo","HeadsUpYesNo","VocalYesNo","ApproachYesNo"]:
        for label, source in [("call",calls),("control",controls)]:
            assert source[field].isin([0,1]).all()
            pairs[f"{field}_{label}"] = source[field]
        a, b = pairs[f"{field}_call"], pairs[f"{field}_control"]
        y = (a-b).to_numpy()
        # Paired risk difference, HC3 SEs. One row per experiment; covariates
        # describe how the within-pair contrast varies across experiments.
        matched = (pairs.Distance==pairs.control_distance) & (pairs.Npenguins==pairs.control_group_size)
        for design, selected in [("all_pairs",np.ones(len(y),dtype=bool)),("matched_conditions",matched)]:
            sub = pairs.loc[selected]
            outcome = y[selected]
            group = (np.log2(sub.Npenguins) + np.log2(sub.control_group_size)) / 2
            context = pd.get_dummies(sub.Context, drop_first=True, dtype=float)
            columns = [np.ones(len(sub)),(sub.fidelity-sub.fidelity.mean())/.1,group,context]
            if design == "all_pairs":
                columns.append(np.log2(sub.Distance/sub.control_distance))
            X = np.column_stack(columns)
            coef = np.linalg.lstsq(X,outcome,rcond=None)[0]
            inv = np.linalg.inv(X.T @ X)
            leverage = np.sum((X @ inv)*X,axis=1)
            score = X*((outcome-X@coef)/(1-leverage))[:,None]
            se = np.sqrt((inv @ score.T @ score @ inv)[1,1])
            margin = t.ppf(.975,len(outcome)-X.shape[1])*se
            adjusted.append(dict(outcome=field,design=design,pairs=len(sub),per_0p1=coef[1],
                                 lower95=coef[1]-margin,upper95=coef[1]+margin))
        for state, sample in [("all",pairs),*pairs.groupby("state",observed=True)]:
            ca, co = sample[f"{field}_call"], sample[f"{field}_control"]
            summary.append(dict(outcome=field,state=state,pairs=len(sample),
                call_events=ca.sum(),control_events=co.sum(),call_rate=ca.mean(),
                control_rate=co.mean(),paired_difference=(ca-co).mean()))
    save(pairs.reset_index(),"playback_pairs")
    save(pd.DataFrame(summary),"playback_summary")
    save(pd.DataFrame(adjusted),"playback_adjusted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default=shutil.which("Rscript"))
    args = parser.parse_args()
    if not args.rscript:
        parser.error("Supply --rscript with the path to Rscript (mgcv required).")
    OUT.mkdir(exist_ok=True)
    values = audit()
    means = np.nanmean(values,axis=2)
    check_missingness(values)
    fits = pooled(means)
    assert (np.diff(fits,axis=-1) <= 1e-12).all()
    table = [dict(call=c,metric=m,d=d,observed=means[i,j,k],fitted=fits[i,j,k])
             for i,c in enumerate(CALLS) for j,m in enumerate(METRICS)
             for k,d in enumerate(DISTANCE)]
    save(pd.DataFrame(table),"channel_curves")
    save(pd.DataFrame([dict(call=c,d=d,observed=fidelity(means)[i,k],fitted=fidelity(fits)[i,k])
                       for i,c in enumerate(CALLS) for k,d in enumerate(DISTANCE)]),"fidelity_knots")
    ranges,sweeps = threshold_analysis(means, fits)
    # Relative arguments keep non-ASCII parent directory names out of Windows Rscript argv.
    r_files = [HERE/"gam_reference.R", OUT/"propagation_long.csv", OUT/"gam_predictions.csv",
               OUT/"audit_63.csv", OUT/"supplementary4_check.csv"]
    rscript = shutil.which(args.rscript) or args.rscript
    subprocess.run([str(Path(rscript).resolve()), "--vanilla",
                    *(os.path.relpath(path, DATA) for path in r_files)],
                   cwd=DATA, check=True)
    gam = pd.read_csv(OUT/"gam_predictions.csv")
    reference = pd.read_csv(OUT/"supplementary4_check.csv")
    compare_models(means, gam)
    playback(fits)
    draw_figures(OUT, DISTANCE, means, fits, sweeps, reference=reference, gam=gam)
    print(ranges.to_string(index=False))


if __name__ == "__main__":
    main()
