"""D. Nested validation, fold predictions and final reference-category contrasts."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acoustics import COLORS, COMMON, LABELS, OUT, SEED, save
from explore import finish, plt


def splits(y, dates, k, grouped):
    # Select the first class-complete partition using labels and dates alone.
    for seed in range(SEED, SEED + 1000):
        splitter = (StratifiedGroupKFold if grouped else StratifiedKFold)(k, shuffle=True, random_state=seed)
        parts = list(splitter.split(np.zeros(len(y)), y, dates if grouped else None))
        if all(len(np.unique(y[idx])) == len(LABELS) for pair in parts for idx in pair):
            if grouped:
                assert all(not set(dates[a]).intersection(dates[b]) for a, b in parts)
            assert np.array_equal(np.sort(np.concatenate([b for a, b in parts])), np.arange(len(y)))
            return parts
    raise ValueError(f"No {k}-fold partition with class coverage. Review grouping.")


def model_specs():
    c_grid = [.1, 1., 10.]
    return {
        "L2": (make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=2000)),
               {"logisticregression__C": c_grid}),
        "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                {"svc__C": c_grid, "svc__gamma": [.05, .2, 1.]}),
        "RF": (RandomForestClassifier(n_estimators=160, random_state=SEED, n_jobs=1),
               {"max_depth": [3, None], "min_samples_leaf": [1, 4]}),
    }


def scores(y, pred):
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred)}


def tuned(spec, x, y, cv):
    model, grid = spec
    return GridSearchCV(clone(model), grid, cv=cv, scoring="f1_macro", error_score="raise").fit(x, y)


def classify(df):
    x = df[COMMON]
    y, dates = df.CallType.to_numpy(), df.Date.to_numpy()
    plans = {}
    for name, grouped in [("date", True), ("stratified", False)]:
        outer = splits(y, dates, 5, grouped)
        plans[name] = [(a, b, splits(y[a], dates[a], 3, grouped)) for a, b in outer]
    records, folds, comparisons, by_class, matrices = [], [], [], [], []
    importance = []
    specs = model_specs()
    for model, spec in specs.items():
        for name, plan in plans.items():
            pred = np.empty(len(y), dtype=object)
            for fold, (a, b, inner) in enumerate(plan, 1):
                fit = tuned(spec, x.iloc[a], y[a], inner)
                pred[b] = fit.predict(x.iloc[b])
                if model == "RF" and name == "date":
                    perm = permutation_importance(fit.best_estimator_, x.iloc[b], y[b], scoring="f1_macro",
                                                  n_repeats=12, random_state=SEED+fold)
                    importance.extend([fold, feature, delta] for feature, delta in zip(COMMON, perm.importances_mean))
                metrics = scores(y[b], pred[b])
                folds.append(dict(model=model, scheme=name, fold=fold, train_n=len(a), test_n=len(b),
                                  inner_best=json.dumps(fit.best_params_), **metrics))
                for pos in b:
                    records.append([model, name, fold, df.index[pos], dates[pos], y[pos], pred[pos]])
            comparisons.append(dict(model=model, scheme=name, **scores(y, pred)))
            p, r, f, n = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
            for i, label in enumerate(LABELS):
                by_class.append([model, name, label, p[i], r[i], f[i], n[i]])
            cm = confusion_matrix(y, pred, labels=LABELS)
            for i, label in enumerate(LABELS):
                matrices.append([model, name, label, *cm[i]])
    folds = save(pd.DataFrame(folds), "fold_scores.csv")
    summary = pd.DataFrame(comparisons).set_index(["model", "scheme"])
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
        g = folds.groupby(["model", "scheme"])[metric]
        summary[metric+"_fold_mean"], summary[metric+"_fold_sd"] = g.mean(), g.std()
    summary = save(summary.reset_index(), "model_comparison.csv")
    save(pd.DataFrame(records, columns=["model", "scheme", "fold", "csv_line", "Date", "true", "pred"]), "oof_predictions.csv")
    classes = save(pd.DataFrame(by_class, columns=["model", "scheme", "CallType", "precision", "recall", "f1", "n"]), "class_scores.csv")
    cms = save(pd.DataFrame(matrices, columns=["model", "scheme", "true", *LABELS]), "confusion_matrices.csv")
    save(pd.DataFrame(importance, columns=["date_fold", "feature", "mean_macro_f1_drop"]), "rf_permutation.csv")
    final = tuned(specs["L2"], x, y, splits(y, dates, 5, True))
    joblib.dump(final.best_estimator_, OUT / "call_classifier.joblib")
    forest = tuned(specs["RF"], x, y, splits(y, dates, 5, True))
    joblib.dump(forest.best_estimator_, OUT / "random_forest.joblib")
    # Post-hoc sensitivity prompted by the small number of correctly identified tv calls.
    weighted_lr = clone(specs["L2"][0]).set_params(logisticregression__class_weight="balanced")
    weighted_pred = np.empty(len(y), dtype=object)
    for a, b, inner in plans["date"]:
        alt = tuned((weighted_lr, specs["L2"][1]), x.iloc[a], y[a], inner)
        weighted_pred[b] = alt.predict(x.iloc[b])
    save(pd.DataFrame({"csv_line": df.index, "true": y, "weighted_L2": weighted_pred}), "weight_sensitivity_predictions.csv")
    sensitivity = scores(y, weighted_pred)
    sensitivity["tv_recall"] = np.mean(weighted_pred[y=="tv"] == "tv")
    save(pd.DataFrame([sensitivity]), "weight_sensitivity.csv")
    lr = final.best_estimator_[-1]
    coeff = []
    for i, lab in enumerate(lr.classes_):
        if lab == "flat":
            continue
        base = list(lr.classes_).index("flat")
        for feature, beta in zip(COMMON, lr.coef_[i] - lr.coef_[base]):
            coeff.append([lab+"/flat", feature, beta, np.exp(beta)])
    coeff = save(pd.DataFrame(coeff, columns=["contrast", "feature", "beta", "odds_ratio"]), "coefficients.csv")
    params = {"features": COMMON, "validation": "date", "seed": SEED, "best_params": final.best_params_,
              "rf_best_params": forest.best_params_, "grids": {m: grid for m, (_, grid) in specs.items()}}
    (OUT / "model_config.json").write_text(json.dumps(params, indent=2), encoding="utf8")
    plot_results(summary, cms, coeff)
    return summary, classes, coeff


def plot_results(summary, cms, coeff):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
    for row, scheme in enumerate(["date", "stratified"]):
        for col, model in enumerate(["L2", "SVM", "RF"]):
            ax = axes[row, col]
            cm = cms[(cms.scheme == scheme) & (cms.model == model)].set_index("true").loc[LABELS, LABELS].to_numpy()
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cms[LABELS].max().max())
            for i, j in np.ndindex(cm.shape):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > .55*cms[LABELS].max().max() else "black")
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
            ax.set(title=f"{model} · {scheme}", xlabel="Predicted", ylabel="Observed")
    finish(fig, "confusion")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained", sharey=True)
    for ax, (scheme, g) in zip(axes, summary.groupby("scheme", sort=False)):
        g.set_index("model")[["accuracy", "macro_f1", "balanced_accuracy"]].plot.bar(ax=ax, color=COLORS, rot=0, legend=False)
        ax.set(title=scheme, ylabel="Pooled out-of-fold score", xlabel="", ylim=(0, 1))
    axes[0].legend(["Accuracy", "Macro-F1", "Balanced accuracy"], fontsize=8)
    finish(fig, "model_comparison")
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    coeff.pivot(index="feature", columns="contrast", values="beta").loc[COMMON].plot.barh(ax=ax, color=COLORS[1:])
    ax.axvline(0, color="black", lw=.7)
    ax.set(xlabel="Change in log odds per 1 SD", ylabel="")
    finish(fig, "coefficients")
