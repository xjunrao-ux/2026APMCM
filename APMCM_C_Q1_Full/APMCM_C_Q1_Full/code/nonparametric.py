"""Reproduce the univariate, permutation and sensitivity analyses for Question 1."""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import anderson_ksamp, kruskal, mannwhitneyu, PermutationMethod, rankdata
from sklearn.decomposition import PCA

from common import COMMON, GROUPS, SEED, holm, load, save_figure, save_table

SECTION = "03_nonparametric"
N_PERM = 49_999
N_RANK_PERM = 19_999
METRICS = (*COMMON, "PulseRate")


def univariate(data):
    pairs, tests = [], []
    rng = np.random.default_rng(SEED)
    for feature in METRICS:
        groups = GROUPS if feature != "PulseRate" else GROUPS[1:]
        samples = [data.loc[data.CallType.eq(g), feature].dropna().to_numpy() for g in groups]
        test = kruskal(*samples) if len(groups) == 3 else mannwhitneyu(*samples)
        ad = anderson_ksamp(samples, variant="midrank", method=PermutationMethod(n_resamples=N_RANK_PERM, rng=rng))
        tests.append({"feature": feature, "test": "KWH" if len(groups) == 3 else "MWU",
                      "groups": "/".join(groups), "n": "/".join(str(len(x)) for x in samples),
                      "statistic": test.statistic, "p": test.pvalue, "AD_statistic": ad.statistic, "AD_p": ad.pvalue})
        if feature == "PulseRate":
            continue
        for (a, x), (b, y) in combinations(zip(groups, samples), 2):
            mw = mannwhitneyu(x, y)
            pairs.append({"feature": feature, "first": a, "second": b, "U": mw.statistic, "p": mw.pvalue,
                          "cliff_delta": 2 * mw.statistic / (len(x) * len(y)) - 1})
    tests = pd.DataFrame(tests)
    tests["p_holm7"] = holm(tests.p)
    tests["AD_p_holm7"] = holm(tests.AD_p)
    pairs = pd.DataFrame(pairs)
    pairs["p_holm18"] = holm(pairs.p)
    save_table(tests, SECTION, "univariate_tests.csv")
    save_table(pairs, SECTION, "exploratory_MWU_pairs.csv")
    return tests


def permutation_indices(n, partitions, rng, draws):
    indices = np.broadcast_to(np.arange(n), (draws, n)).copy()
    for positions in partitions:
        indices[:, positions] = rng.permuted(indices[:, positions], axis=1)
    return indices


def blocked_rank(data):
    rng, records = np.random.default_rng(SEED), []
    for feature in METRICS:
        rows = data.dropna(subset=[feature]).reset_index(drop=True)
        _, codes, counts = np.unique(rows.CallType, return_inverse=True, return_counts=True)
        membership = np.eye(len(counts))[codes]
        ranks = rankdata(rows[feature])
        partitions = [idx for idx in rows.groupby(["Date", "Location"]).indices.values()
                      if np.unique(codes[idx]).size > 1]
        observed = (ranks @ membership)**2 @ (1 / counts)
        shuffled = ranks[permutation_indices(len(rows), partitions, rng, N_RANK_PERM)]
        null = (shuffled @ membership)**2 @ (1 / counts)
        records.append({"feature": feature, "p": (1 + (null >= observed - 1e-8).sum()) / (N_RANK_PERM + 1)})
    result = pd.DataFrame(records)
    result["p_holm7"] = holm(result.p)
    save_table(result, SECTION, "blocked_rank_sensitivity.csv")


def permanova(x, labels, rng, blocks=None):
    n = len(x)
    _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    design = np.eye(len(counts))[codes, :-1]
    partitions = [np.arange(n)] if blocks is None else [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    y = x.copy()
    for idx in partitions:
        design[idx] -= design[idx].mean(axis=0)
        y[idx] -= y[idx].mean(axis=0)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    model_df = int((singular > 1e-10).sum())
    residual_df = n - model_df - len(partitions)
    q = u[:, :model_df]
    total = (y**2).sum()
    fit = ((q.T @ y)**2).sum()
    residual = total - fit
    if blocks is None:
        np.testing.assert_allclose(total, (pdist(x)**2).sum() / n)
        np.testing.assert_allclose(residual, sum((pdist(x[codes == c])**2).sum() / size
                                                 for c, size in enumerate(counts)))
    moving = [idx for idx in partitions if np.unique(codes[idx]).size > 1]
    hit = 0
    for offset in range(0, N_PERM, 1024):
        draws = min(1024, N_PERM - offset)
        index = permutation_indices(n, moving, rng, draws)
        null = ((q.T @ y[index])**2).sum(axis=(1, 2))
        hit += (null >= fit - 1e-10).sum()
    return {"pseudo_F": (fit / model_df) / (residual / residual_df),
            "R2": fit / total, "p": (1 + hit) / (1 + N_PERM),
            "df_model": model_df, "df_residual": residual_df}


def dispersion(x, labels, rng):
    _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n, k = len(codes), len(counts)
    membership = np.eye(k)[codes]
    distances = np.zeros(n)
    fitted = np.zeros(n)
    for c, size in enumerate(counts):
        is_group = codes == c
        points = x[is_group]

        def loss(center):
            return np.linalg.norm(points - center, axis=1).sum()

        def gradient(center):
            offset = center - points
            norms = np.linalg.norm(offset, axis=1)
            return (offset / np.maximum(norms, 1e-13)[:, None]).sum(axis=0)

        median = minimize(loss, points.mean(axis=0), jac=gradient, method="BFGS", options={"gtol": 1e-9})
        assert np.linalg.norm(median.jac) < 1e-5
        distances[is_group] = np.linalg.norm(points - median.x, axis=1) * np.sqrt(size / (size - 1))
        fitted[is_group] = distances[is_group].mean()
    residuals = distances - fitted
    ss_error = (residuals**2).sum()
    f = ((fitted - distances.mean())**2).sum() / (k - 1) / (ss_error / (n - k))
    hit = 0
    for offset in range(0, N_PERM, 1024):
        draws = min(1024, N_PERM - offset)
        index = permutation_indices(n, [np.arange(n)], rng, draws)
        ss = (residuals[index] @ membership)**2 @ (1 / counts)
        f_null = (ss / (k - 1)) / ((ss_error - ss) / (n - k))
        hit += (f_null >= f - 1e-10).sum()
    return {"dispersion_F": f, "dispersion_p": (1 + hit) / (1 + N_PERM)}


def multivariate(data):
    rng = np.random.default_rng(SEED)
    raw = data[list(COMMON)]
    z = ((raw - raw.mean()) / raw.std()).to_numpy()
    labels = data.CallType.to_numpy()
    blocks = data.groupby(["Date", "Location"], sort=False).ngroup().to_numpy()
    false_bay = data.Location.eq("FalseBay")
    pca = PCA(svd_solver="full").fit(z)
    sign = np.sign(pca.components_[0, COMMON.index("Q50")])
    pc1 = pca.transform(z)[:, 0] * sign
    overall = [{"comparison": "all", **permanova(z, labels, rng), **dispersion(z, labels, rng)},
               {"comparison": "all_blocked", **permanova(z, labels, rng, blocks)},
               {"comparison": "without_Fpeak", **permanova(z[:, [0, 1, 3, 4, 5]], labels, rng)},
               {"comparison": "FalseBay_only", **permanova(z[false_bay], labels[false_bay], rng)}]
    pairwise = []
    for a, b in combinations(GROUPS, 2):
        keep = np.isin(labels, [a, b])
        fit = permanova(z[keep], labels[keep], rng)
        blocked = permanova(z[keep], labels[keep], rng, blocks[keep])
        d = dispersion(z[keep], labels[keep], rng)
        pc = mannwhitneyu(pc1[labels == a], pc1[labels == b])
        pairwise.append({"first": a, "second": b, **fit, "blocked_R2": blocked["R2"], "blocked_p": blocked["p"],
                         **d, "PC1_U": pc.statistic, "PC1_p": pc.pvalue})
    pairs = pd.DataFrame(pairwise)
    for col in ("p", "blocked_p", "dispersion_p", "PC1_p"):
        pairs[f"{col}_holm3"] = holm(pairs[col])
    save_table(pd.DataFrame(overall), SECTION, "PERMANOVA_overall.csv")
    save_table(pairs, SECTION, "PERMANOVA_pairs.csv")
    save_table(pd.DataFrame({"feature": COMMON, "PC1_weight": pca.components_[0] * sign,
                             "PC1_variance_fraction": pca.explained_variance_ratio_[0]}), SECTION, "PCA_PC1.csv")


def plot(tests):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    x = np.arange(len(tests))
    ax.bar(x - .18, -np.log10(tests.p), .35, label="Raw p", color="#56B4E9")
    ax.bar(x + .18, -np.log10(tests.p_holm7), .35, label="Holm-adjusted p", color="#0072B2")
    ax.axhline(-np.log10(.05), ls="--", lw=1, color="#D55E00", label="p = 0.05")
    ax.set(xticks=x, xticklabels=tests.feature, ylabel="−log₁₀(p)", title="Tests by acoustic feature")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, SECTION, "univariate_p_values")


def main():
    data = load("Data01_clean.csv")
    tests = univariate(data)
    blocked_rank(data)
    multivariate(data)
    plot(tests)


if __name__ == "__main__":
    main()
