"""Train, validate, or apply the three-stage Data01 classifier."""

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
CLASSES = ["flat", "mod", "tv"]
PARTS = {
    "duration": ["Duration"],
    "spectrum": ["F01", "Fpeak", "Q25", "Q50", "Q75"],
}
FEATURES = PARTS["duration"] + PARTS["spectrum"]
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = np.linspace(DEFAULT_THRESHOLD, 1.0, 11)
METHODS = ("rules_only", "single_logistic_05", "stack_05", "stack_tuned")


def route(rows):
    f02 = pd.to_numeric(rows["F02"])
    pulse = pd.to_numeric(rows["PulseRate"])
    for values in (f02, pulse):
        measured = values.dropna().to_numpy()
        if not np.isfinite(measured).all() or (measured <= 0).any():
            raise ValueError(f"{values.name} must contain positive finite values or blanks.")
    stages = pd.Series(3, index=rows.index)
    stages.loc[pulse.notna()] = 2
    stages.loc[f02.notna()] = 1
    return stages


def grouped_folds(rows):
    groups = rows["Date"].astype(str) + "|" + rows["Location"].astype(str)
    return LeaveOneGroupOut().split(rows, rows["CallType"], groups)


def logistic():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    )


def fit_base(rows):
    return {
        name: logistic().fit(rows[columns], rows["CallType"])
        for name, columns in PARTS.items()
    }


def base_scores(models, rows):
    return pd.DataFrame({
        name: models[name].predict_proba(rows[columns])[:, 1]
        for name, columns in PARTS.items()
    }, index=rows.index)


def fit_stack(rows):
    base_oof = pd.DataFrame(index=rows.index, columns=list(PARTS), dtype=float)
    for training, test in grouped_folds(rows):
        base = fit_base(rows.iloc[training])
        base_oof.iloc[test] = base_scores(base, rows.iloc[test])
    meta = logistic().fit(base_oof, rows["CallType"])
    return {"base": fit_base(rows), "meta": meta}


def stack_score(model, rows):
    return model["meta"].predict_proba(base_scores(model["base"], rows))[:, 1]


def fit_threshold(rows):
    oof = np.empty(len(rows))
    for training, test in grouped_folds(rows):
        model = fit_stack(rows.iloc[training])
        oof[test] = stack_score(model, rows.iloc[test])
    actual = rows["CallType"].eq("mod").to_numpy()

    def criterion(cutoff):
        predicted = oof > cutoff
        # Prefer more recovered calls at equal accuracy, then the higher threshold.
        return (np.sum(predicted == actual), np.sum(predicted & actual), cutoff)

    return float(max(THRESHOLDS, key=criterion))


def evaluate(rows):
    stages = route(rows)
    blank = stages.eq(3)
    predictions = rows[["source_row", "Date", "Location", "CallType", "F02", "PulseRate"]].copy()
    predictions["stage"] = stages
    predictions["rules_only"] = np.select(
        [stages.eq(1), stages.eq(2)], ["tv", "mod"], default="flat"
    )
    for method in METHODS[1:]:
        predictions[method] = predictions["rules_only"]
    for fold, (training, test) in enumerate(grouped_folds(rows), start=1):
        train_rows = rows.iloc[training]
        test_rows = rows.iloc[test]
        predictions.loc[test_rows.index, "fold"] = fold
        blank_test = test_rows.loc[blank.loc[test_rows.index]]
        if blank_test.empty:
            continue
        blank_train = train_rows.loc[blank.loc[train_rows.index]]
        stack = fit_stack(blank_train)
        cutoff = fit_threshold(blank_train)
        score = stack_score(stack, blank_test)
        direct = logistic().fit(blank_train[FEATURES], blank_train["CallType"])
        direct_score = direct.predict_proba(blank_test[FEATURES])[:, 1]
        idx = blank_test.index
        predictions.loc[idx, "stack_score_mod"] = score
        predictions.loc[idx, "stack_threshold"] = cutoff
        predictions.loc[idx, "single_score_mod"] = direct_score
        predictions.loc[idx, "single_logistic_05"] = np.where(
            direct_score > DEFAULT_THRESHOLD, "mod", "flat"
        )
        predictions.loc[idx, "stack_05"] = np.where(score > DEFAULT_THRESHOLD, "mod", "flat")
        predictions.loc[idx, "stack_tuned"] = np.where(score > cutoff, "mod", "flat")
    predictions["fold"] = predictions["fold"].astype(int)
    return predictions


def summarize(predictions):
    result = []
    actual = predictions["CallType"]
    blank = predictions["stage"].eq(3)
    for method in METHODS:
        predicted = predictions[method]
        result.append({
            "method": method,
            "overall_correct": int(predicted.eq(actual).sum()),
            "overall_accuracy": predicted.eq(actual).mean(),
            "overall_macro_f1": f1_score(actual, predicted, labels=CLASSES, average="macro"),
            "stage3_correct": int((predicted.eq(actual) & blank).sum()),
            "stage3_accuracy": predicted[blank].eq(actual[blank]).mean(),
            "blank_mod_recovered": int((blank & actual.eq("mod") & predicted.eq("mod")).sum()),
            "flat_false_positives": int((blank & actual.eq("flat") & predicted.eq("mod")).sum()),
        })
    return pd.DataFrame(result)


def fit(data_path, output):
    rows = pd.read_csv(data_path, encoding="utf-8-sig")
    stages = route(rows)
    blanks = rows.loc[stages.eq(3)]
    if not np.isfinite(blanks[FEATURES].to_numpy()).all():
        raise ValueError("Stage three requires six finite features.")
    predictions = evaluate(rows)
    metrics = summarize(predictions)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    missing_mod = predictions.loc[predictions["stage"].eq(3) & predictions["CallType"].eq("mod")]
    missing_mod.to_csv(output / "unmeasured_pulse_mod.csv", index=False, encoding="utf-8-sig")
    model = {"stack": fit_stack(blanks), "threshold": DEFAULT_THRESHOLD}
    joblib.dump(model, output / "models.joblib", compress=3)
    print(metrics.round(4).to_string(index=False))
    print(missing_mod[["source_row", "Date", "stack_score_mod", "stack_threshold",
                       "stack_05", "stack_tuned", "single_logistic_05"]].to_string(index=False))
    print(f"Prediction threshold: {model['threshold']:.2f}")


def predict(data_path, model_path, output, threshold):
    rows = pd.read_csv(data_path, encoding="utf-8-sig")
    stages = route(rows)
    model = joblib.load(model_path)
    cutoff = model["threshold"] if threshold is None else threshold
    rows["classification_stage"] = stages
    rows["predicted_CallType"] = np.select(
        [stages.eq(1), stages.eq(2)], ["tv", "mod"], default="flat"
    )
    blank = stages.eq(3)
    if blank.any():
        score = stack_score(model["stack"], rows.loc[blank])
        rows.loc[blank, "stage3_score_mod"] = score
        rows.loc[blank, "predicted_CallType"] = np.where(score > cutoff, "mod", "flat")
    rows.to_csv(output, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=HERE / "Data01_clean.csv")
    train.add_argument("--output", type=Path, default=HERE / "results")
    inference = commands.add_parser("predict")
    inference.add_argument("data", type=Path)
    inference.add_argument("--model", type=Path, default=HERE / "results" / "models.joblib")
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.command == "train":
        fit(args.data, args.output)
    else:
        if args.threshold is not None and not 0 <= args.threshold <= 1:
            parser.error("--threshold must be between 0 and 1")
        predict(args.data, args.model, args.output, args.threshold)


if __name__ == "__main__":
    main()
