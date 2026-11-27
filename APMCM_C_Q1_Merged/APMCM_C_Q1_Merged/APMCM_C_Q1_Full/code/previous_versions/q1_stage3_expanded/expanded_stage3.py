"""Evaluate and apply three-stage classifiers trained on all flat and mod calls."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SPECTRUM = ["F01", "Fpeak", "Q25", "Q50", "Q75"]
COMMON = ["Duration", *SPECTRUM]
PULSE_TERMS = ["pulse_observed", "pulse_centered_if_observed"]
THRESHOLD = 0.5
VARIANTS = {
    "blank6_stack": (False, True),
    "all6_stack": (False, True),
    "all7_stack": (True, True),
    "all6_linear": (False, False),
    "all7_linear": (True, False),
}
DEPLOYABLE = tuple(name for name in VARIANTS if name.startswith("all"))


def route(rows):
    f02 = pd.to_numeric(rows["F02"])
    pulse = pd.to_numeric(rows["PulseRate"])
    for values in (f02, pulse):
        observed = values.dropna().to_numpy()
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError(f"{values.name} must contain positive finite measurements or blanks.")
    return pd.Series(
        np.select([f02.notna(), pulse.notna()], [1, 2], default=3), index=rows.index
    )


def grouped_folds(rows):
    groups = rows["Date"].astype(str) + "|" + rows["Location"].astype(str)
    return LeaveOneGroupOut().split(rows, rows["CallType"], groups)


def logistic():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    )


def design(rows, pulse_center):
    features = rows[COMMON].apply(pd.to_numeric)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Logistic regression requires six finite common features.")
    if pulse_center is not None:
        pulse = pd.to_numeric(rows["PulseRate"])
        observed = pulse.notna()
        # Zero is the contribution of an inactive term, not an imputed pulse rate.
        features[PULSE_TERMS[0]] = observed.astype(float)
        features[PULSE_TERMS[1]] = (pulse - pulse_center).where(observed, 0.0)
    return features


def fit_base(rows, include_pulse, stacked):
    center = float(rows["PulseRate"].dropna().mean()) if include_pulse else None
    if include_pulse and not np.isfinite(center):
        raise ValueError("The seven-feature model needs some observed pulse rates in training.")
    pulse_terms = PULSE_TERMS if include_pulse else []
    blocks = (
        {"duration": ["Duration"], "spectrum": SPECTRUM + pulse_terms}
        if stacked else {"linear": COMMON + pulse_terms}
    )
    features = design(rows, center)
    models = {
        name: logistic().fit(features[columns], rows["CallType"])
        for name, columns in blocks.items()
    }
    return {"center": center, "blocks": blocks, "models": models}


def block_scores(base, rows):
    features = design(rows, base["center"])
    return pd.DataFrame({
        name: base["models"][name].predict_proba(features[columns])[:, 1]
        for name, columns in base["blocks"].items()
    }, index=rows.index)


def fit_model(rows, include_pulse, stacked):
    base = fit_base(rows, include_pulse, stacked)
    if not stacked:
        return {"base": base, "meta": None}

    scores = pd.DataFrame(index=rows.index, columns=list(base["blocks"]), dtype=float)
    for training, test in grouped_folds(rows):
        inner_model = fit_base(rows.iloc[training], include_pulse, stacked)
        scores.iloc[test] = block_scores(inner_model, rows.iloc[test])
    meta = logistic().fit(scores, rows["CallType"])
    return {"base": base, "meta": meta}


def mod_score(model, rows):
    scores = block_scores(model["base"], rows)
    if model["meta"] is None:
        return scores["linear"].to_numpy()
    return model["meta"].predict_proba(scores)[:, 1]


def class_from_score(score):
    return np.where(score > THRESHOLD, "mod", "flat")


def inspect_metrics(result):
    metrics = []
    for method in ("rules", *VARIANTS):
        scopes = {
            "stage3": result.loc[result["stage"] == 3],
            "cascade": result,
        }
        if method in DEPLOYABLE:
            scopes["flat_mod_without_routing"] = result.loc[result["CallType"].isin(["flat", "mod"])]
        for scope, cases in scopes.items():
            label = cases["CallType"]
            pred = cases[f"{method}_direct"] if scope == "flat_mod_without_routing" else cases[method]
            mod = label.eq("mod")
            flat = label.eq("flat")
            metrics.append({
                "model": method, "scope": scope, "n": len(cases),
                "n_correct": int(pred.eq(label).sum()),
                "accuracy": pred.eq(label).mean(),
                "macro_f1": f1_score(label, pred, average="macro"),
                "mod_recall": pred[mod].eq("mod").mean(),
                "flat_recall": pred[flat].eq("flat").mean(),
                "blank_mod_recovered": int((mod & cases["stage"].eq(3) & pred.eq("mod")).sum()),
                "flat_false_positives": int((flat & pred.eq("mod")).sum()),
            })
    return pd.DataFrame(metrics)


def run_experiment(rows):
    stages = route(rows)
    flat_mod = rows["CallType"].isin(["flat", "mod"])
    blank = stages.eq(3)
    if not stages.eq(1).equals(rows["CallType"].eq("tv")) or not stages[flat_mod].isin([2, 3]).all():
        raise ValueError("This comparison expects the original Data01 label and routing structure.")
    result = rows[["source_row", "Date", "Location", "CallType", "F02", "PulseRate", *COMMON]].copy()
    result["stage"] = stages
    result["rules"] = np.select([stages.eq(1), stages.eq(2)], ["tv", "mod"], default="flat")
    for name in VARIANTS:
        result[name] = result["rules"]

    for fold, (train, test) in enumerate(grouped_folds(rows), start=1):
        train_idx = rows.index[train]
        test_idx = rows.index[test]
        result.loc[test_idx, "fold"] = fold
        for name, settings in VARIANTS.items():
            eligible = blank if name == "blank6_stack" else flat_mod
            test_subset = test_idx[eligible.loc[test_idx]]
            if test_subset.empty:
                continue
            train_subset = train_idx[eligible.loc[train_idx]]
            model = fit_model(rows.loc[train_subset], *settings)
            score = mod_score(model, rows.loc[test_subset])
            result.loc[test_subset, f"{name}_score_mod"] = score
            result.loc[test_subset, f"{name}_direct"] = class_from_score(score)
            route3 = test_subset[blank.loc[test_subset]]
            result.loc[route3, name] = result.loc[route3, f"{name}_direct"]

    result["fold"] = result["fold"].astype(int)
    return result


def train(data_path, output):
    rows = pd.read_csv(data_path, encoding="utf-8-sig")
    predictions = run_experiment(rows)
    metrics = inspect_metrics(predictions)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    missed_branch = predictions["stage"].eq(3) & predictions["CallType"].eq("mod")
    predictions.loc[missed_branch].to_csv(
        output / "pulse_blank_mod.csv", index=False, encoding="utf-8-sig"
    )
    pooled = rows.loc[rows["CallType"].isin(["flat", "mod"])]
    models = {name: fit_model(pooled, *VARIANTS[name]) for name in DEPLOYABLE}
    joblib.dump(models, output / "models.joblib", compress=3)
    print(metrics.round(4).to_string(index=False))
    columns = ["source_row", "blank6_stack", *DEPLOYABLE]
    print(predictions.loc[missed_branch, columns].to_string(index=False))


def predict(data_path, output, model_path, variant):
    rows = pd.read_csv(data_path, encoding="utf-8-sig")
    stages = route(rows)
    rows["classification_stage"] = stages
    rows["predicted_CallType"] = np.select(
        [stages.eq(1), stages.eq(2)], ["tv", "mod"], default="flat"
    )
    rows["stage3_score_mod"] = np.nan
    blank = stages.eq(3)
    if blank.any():
        model = joblib.load(model_path)[variant]
        score = mod_score(model, rows.loc[blank])
        rows.loc[blank, "stage3_score_mod"] = score
        rows.loc[blank, "predicted_CallType"] = class_from_score(score)
    rows.to_csv(output, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("train")
    fit_parser.add_argument("--data", type=Path, default=HERE / "Data01_clean.csv")
    fit_parser.add_argument("--output", type=Path, default=HERE / "results")
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("data", type=Path)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--models", type=Path, default=HERE / "results" / "models.joblib")
    predict_parser.add_argument("--variant", choices=DEPLOYABLE, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output)
    else:
        predict(args.data, args.output, args.models, args.variant)


if __name__ == "__main__":
    main()
