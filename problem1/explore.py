"""B. Describe the calls. C. Compare their rank distributions."""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from acoustics import COLORS, COMMON, FIG, LABELS, MEASURES, SEED, UNITS, report, save, table


def finish(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def explore(df):
    rows = []
    for label, block in df.groupby("CallType"):
        for col in MEASURES:
            x = block[col].dropna()
            q1, q3 = x.quantile([.25, .75])
            rows.append([label, col, len(x), len(block)-len(x), x.mean(), x.std(),
                         x.min(), q1, x.median(), q3, q3-q1, x.max()])
    desc = save(pd.DataFrame(rows, columns=["CallType", "feature", "n", "missing", "mean", "sd",
                                            "min", "q1", "median", "q3", "IQR", "max"]), "descriptive.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3), layout="constrained")
    counts = df.CallType.value_counts().reindex(LABELS)
    ax.bar_label(ax.bar(LABELS, counts, color=COLORS))
    ax.set(ylabel="Calls", ylim=(0, counts.max()*1.15))
    finish(fig, "sample_counts")
    for kind in ["box", "distribution"]:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
        for ax, col in zip(axes.flat, MEASURES):
            bins = np.histogram_bin_edges(df[col].dropna(), bins="sturges")
            for i, (label, color) in enumerate(zip(LABELS, COLORS)):
                x = df.loc[df.CallType.eq(label), col].dropna()
                if x.empty:
                    continue
                if kind == "box":
                    ax.boxplot(x, positions=[i], widths=.56, patch_artist=True,
                               boxprops={"facecolor": color, "alpha": .7}, medianprops={"color": "black"})
                else:
                    ax.hist(x, bins=bins, histtype="step", color=color, linewidth=1.8, label=label)
            ax.set_title(f"{col} ({UNITS[col]})")
            if kind == "box":
                ax.set_xticks(range(3), LABELS)
                ax.set_xlim(-.6, 2.6)
            else:
                ax.set_ylabel("Calls")
        if kind == "distribution":
            axes[0, 0].legend(frameon=False)
        finish(fig, kind)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, method in zip(axes, ["pearson", "spearman"]):
        corr = df[COMMON].corr(method=method)
        save(corr.rename_axis("feature").reset_index(), f"{method}.csv")
        img = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(COMMON)), COMMON, rotation=45, ha="right")
        ax.set_yticks(range(len(COMMON)), COMMON)
        ax.set_title(method.title())
        for i, j in np.ndindex(corr.shape):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if abs(v)>.64 else "black")
    fig.colorbar(img, ax=axes, shrink=.8)
    finish(fig, "correlations")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for ax, (a, b) in zip(axes, [("Duration", "F01"), ("F01", "Q50"), ("Q25", "Q75")]):
        for lab, color in zip(LABELS, COLORS):
            g = df[df.CallType.eq(lab)]
            ax.scatter(g[a], g[b], s=18, color=color, alpha=.7, label=lab)
        ax.set(xlabel=f"{a} ({UNITS[a]})", ylabel=f"{b} ({UNITS[b]})")
    axes[0].legend(frameon=False)
    finish(fig, "scatter")
    pearson = df[COMMON].corr()
    vif = save(pd.DataFrame({"feature": COMMON, "VIF": np.diag(np.linalg.inv(pearson))}), "vif.csv")
    pair = pearson.abs().where(np.triu(np.ones(pearson.shape), 1).astype(bool)).stack().idxmax()
    report("eda_report.md", f"""# B · 声学分布

各组中位数见下表，空格用破折号表示无观测。descriptive.csv 给出每格有效数、缺失数、均值、样本标准差、四分位数、IQR 和范围。

{table(desc.pivot(index="feature", columns="CallType", values="median").reset_index())}

类别分布及重叠见 figures/box.png、distribution.png 与 scatter.png。直方图在同一特征内共用分箱，纵轴为观测数。时长结合第一基频描述时间与发声结构，三个频谱分位数共同描述能量分布。

六变量中 Pearson 相关绝对值最大的一对为 {pair[0]} 与 {pair[1]}（r = {pearson.loc[pair]:.3f}）。Spearman 结果见 spearman.csv。VIF 由含截距回归所对应的相关矩阵逆矩阵对角线计算。分位频率间的共享信息会影响多变量系数，解释时应同时参考类内分布和正则化强度。

{table(vif)}
""")
    return desc


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    adj[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adj


def blocked_pvalues(df, repetitions=4999):
    """Conditional null: permute call labels within date, preserving each day's composition."""
    ranks = df[COMMON].rank().to_numpy()
    y = pd.Categorical(df.CallType, categories=LABELS).codes.copy()
    groups = list(df.groupby("Date").indices.values())
    def statistic(labels):
        sums = np.stack([ranks[labels == i].sum(axis=0) for i in range(len(LABELS))])
        return (sums**2 / np.bincount(labels)[:, None]).sum(axis=0)
    observed = statistic(y)
    rng = np.random.default_rng(SEED)
    exceed = np.zeros(len(COMMON), dtype=int)
    perm = y.copy()
    for _ in range(repetitions):
        for idx in groups:
            perm[idx] = rng.permutation(y[idx])
        exceed += statistic(perm) >= observed - 1e-8
    return (exceed + 1) / (repetitions + 1)


def tests(df):
    rows = []
    for col in COMMON:
        xs = [df.loc[df.CallType.eq(label), col] for label in LABELS]
        h, p = stats.kruskal(*xs)
        rows.append([col, "Kruskal-Wallis", "flat/mod/tv", h, p, "epsilon2_rank", h/(len(df)-1), "omnibus_6"])
    for col in COMMON + ["PulseRate"]:
        pairs = combinations(LABELS, 2) if col in COMMON else [("mod", "tv")]
        for a, b in pairs:
            x, y = [df.loc[df.CallType.eq(lab), col].dropna() for lab in (a, b)]
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            rows.append([col, "Mann-Whitney", f"{a}/{b}", u, p, "Cliff_delta",
                         2*u/(len(x)*len(y))-1, "posthoc_18" if col in COMMON else "pulse_1"])
    result = pd.DataFrame(rows, columns=["feature", "test", "groups", "statistic", "p_raw", "effect_name", "effect", "family"])
    result["p_holm"] = result.groupby("family").p_raw.transform(holm)
    for i, row in result.iterrows():
        for label in row.groups.split("/"):
            x = df.loc[df.CallType.eq(label), row.feature].dropna()
            result.loc[i, [label+"_n", label+"_median", label+"_IQR"]] = [len(x), x.median(), x.quantile(.75)-x.quantile(.25)]
    save(result, "nonparametric_tests.csv")
    p = blocked_pvalues(df)
    save(pd.DataFrame({"feature": COMMON, "p_within_date": p, "p_holm": holm(p)}), "within_date_permutation.csv")
    return result
