"""Reproduce the staged classifier and the previously compared feature subsets."""
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from common import GROUPS, ROOT, load, save_figure, save_table

ARCHIVE = ROOT / "code" / "previous_versions" / "q1_stage3_expanded"
sys.path.insert(0, str(ARCHIVE))
import expanded_stage3 as existing

SECTION = "02_classification"
FEATURE_SETS = {
    "Fpeak": ["Fpeak"],
    "Fpeak_F01": ["Fpeak", "F01"],
    "Fpeak_Q50": ["Fpeak", "Q50"],
    "Fpeak_F01_Q50": ["Fpeak", "F01", "Q50"],
    "Fpeak_pulse_flag": ["Fpeak", existing.PULSE_TERMS[0]],
    "Fpeak_pulse_full": ["Fpeak", *existing.PULSE_TERMS],
}


def add_feature_experiments(rows, result, eligible):
    for name in FEATURE_SETS:
        result[name] = result["rules"]
    for train, test in existing.grouped_folds(rows):
        train = train[eligible.iloc[train]]
        test = test[eligible.iloc[test]]
        if not len(test):
            continue
        training = rows.iloc[train]
        testing = rows.iloc[test]
        center = training.PulseRate.dropna().mean()
        design_train = existing.design(training, center)
        design_test = existing.design(testing, center)
        for name, columns in FEATURE_SETS.items():
            model = existing.logistic().fit(design_train[columns], training.CallType)
            score = model.predict_proba(design_test[columns])[:, 1]
            result.loc[testing.index, f"{name}_score_mod"] = score
            result.loc[testing.index, f"{name}_direct"] = existing.class_from_score(score)
            routed = testing.index[result.loc[testing.index, "stage"].eq(3)]
            result.loc[routed, name] = result.loc[routed, f"{name}_direct"]


def summarize(result, eligible):
    records = []
    stage = result.stage.eq(3)
    for name in ("rules", *existing.VARIANTS, *FEATURE_SETS):
        has_direct = name in (*existing.DEPLOYABLE, *FEATURE_SETS)
        prediction = result[name]
        stage_true = result.loc[stage, "CallType"]
        stage_pred = prediction[stage]
        record = {"model": name, "n_direct": 84 if has_direct or name == "rules" else None,
                  "direct_correct": None, "direct_auc": None,
                  "stage3_correct": int(stage_pred.eq(stage_true).sum()),
                  "blank_mod_recovered": int((stage_true.eq("mod") & stage_pred.eq("mod")).sum()),
                  "flat_false_positives": int((stage_true.eq("flat") & stage_pred.eq("mod")).sum()),
                  "stage3_balanced_accuracy": balanced_accuracy_score(stage_true, stage_pred),
                  "cascade_correct": int(result.CallType.eq(prediction).sum())}
        if has_direct or name == "rules":
            true = result.loc[eligible, "CallType"]
            direct = result.loc[eligible, f"{name}_direct"] if has_direct else prediction[eligible]
            scores = result.loc[eligible, f"{name}_score_mod"] if has_direct else direct.eq("mod").astype(float)
            record["direct_correct"] = direct.eq(true).sum()
            record["direct_auc"] = roc_auc_score(true.eq("mod"), scores)
        records.append(record)
    return pd.DataFrame(records)


def plot(results, matrix):
    fig, ax = plt.subplots(figsize=(5.2, 4.5), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > 25 else "#1A365D", fontsize=13)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=GROUPS, yticklabels=GROUPS,
           xlabel="Predicted type", ylabel="Observed type", title="Out-of-fold cascade · 50-sample stack")
    fig.colorbar(image, ax=ax, shrink=.7, label="Calls")
    save_figure(fig, SECTION, "cascade_confusion")
    selected = results.set_index("model").loc[["rules", "blank6_stack", "Fpeak", "Fpeak_F01",
                                               "Fpeak_Q50", "Fpeak_F01_Q50", "Fpeak_pulse_full",
                                               "all6_linear", "all7_linear"]]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    y = np.arange(len(selected))
    ax.barh(y - .18, selected.blank_mod_recovered, .32, label="Modulated calls recovered (of 6)", color="#0072B2")
    ax.barh(y + .18, selected.flat_false_positives, .32, label="Flat calls misclassified (of 44)", color="#D55E00")
    ax.set(yticks=y, yticklabels=[name.replace("_", " ") for name in selected.index], xlabel="Number of calls",
           title="What reached stage 3")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, SECTION, "stage3_tradeoff")


def main():
    rows = load("Data01_clean.csv")
    eligible = rows.CallType.isin(["flat", "mod"])
    result = existing.run_experiment(rows)
    add_feature_experiments(rows, result, eligible)
    archive_original = ROOT / "code/previous_versions/q1_three_stage/results/oof_predictions.csv"
    original = pd.read_csv(archive_original).set_index("source_row").loc[result.source_row]
    assert np.array_equal(result.blank6_stack, original.stack_05)
    result["original_single_05"] = original.single_logistic_05.to_numpy()
    result["original_stack_tuned"] = original.stack_tuned.to_numpy()
    stats = summarize(result, eligible)
    matrix = confusion_matrix(result.CallType, result.blank6_stack, labels=GROUPS)
    save_table(stats, SECTION, "classification_metrics.csv")
    save_table(result, SECTION, "classification_oof.csv")
    save_table(result[result.stage.eq(3) & result.CallType.eq("mod")], SECTION, "six_hard_mod_cases.csv")
    save_table(pd.DataFrame(matrix, index=pd.Index(GROUPS, name="actual"), columns=GROUPS).reset_index(),
               SECTION, "cascade_confusion.csv")
    plot(stats, matrix)
    model = {"stack": existing.fit_model(rows[result.stage.eq(3)], include_pulse=False, stacked=True),
             "threshold": existing.THRESHOLD}
    joblib.dump(model, ROOT / SECTION / "models" / "cascade_50_stage3.joblib", compress=3)


if __name__ == "__main__":
    main()
