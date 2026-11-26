# -*- coding: utf-8 -*-
"""第一问非参数检验：PERMANOVA（多元）+ Mann-Whitney U（单变量两两）。
特征集：六项共同声学指标 Duration/F01/Fpeak/Q25/Q50/Q75（三类均有定义）。
PERMANOVA：z-score 后欧氏距离，9999 次置换；全局 + 三对两两(Holm 校正)。
MWU：每特征三对两两(18 项，Holm 校正) + PulseRate(mod vs tv, 单项)。
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

DATA = Path(r"G:\亚太数模\2026APMCM\题目与要求\date\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
FEAT6 = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
LABELS = ["flat", "mod", "tv"]
NPERM = 9999
SEED = 2026


def pseudo_F_and_r2(D, labels):
    """Anderson(2001) PERMANOVA pseudo-F 与 R^2（标签为 0..a-1 整数）。"""
    labels = np.asarray(labels)
    tri = squareform(D)
    N = len(labels)
    SST = np.sum(np.triu(tri ** 2, 1)) / N
    SSW = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        sub = tri[np.ix_(idx, idx)]
        SSW += np.sum(np.triu(sub ** 2, 1)) / len(idx)
    a = len(np.unique(labels))
    SSA = SST - SSW
    if SSW == 0:
        F = np.inf
    else:
        F = (SSA / (a - 1)) / (SSW / (N - a))
    return F, SSA / SST if SST > 0 else np.nan


def permanova(D, labels, nperm=NPERM, rng=None):
    """返回 (F_obs, R2_obs, p_value)。"""
    rng = rng or np.random.default_rng(SEED)
    labels = np.asarray(labels)
    F_obs, R2 = pseudo_F_and_r2(D, labels)
    exceed = 0
    for _ in range(nperm):
        perm = rng.permutation(labels)
        Fp, _ = pseudo_F_and_r2(D, perm)
        if Fp >= F_obs:
            exceed += 1
    return F_obs, R2, (exceed + 1) / (nperm + 1)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    adj[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adj


def main():
    df = pd.read_csv(DATA)
    df["CallType"] = df.CallType.str.lower()

    # z-score（检验用全量标准化，不涉及 train/test 划分）
    Z = (df[FEAT6] - df[FEAT6].mean()) / df[FEAT6].std()
    D = pdist(Z.values, metric="euclidean")  # condensed 距离矩阵
    codes = df.CallType.map({l: i for i, l in enumerate(LABELS)}).to_numpy()

    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("PERMANOVA（z-score 后欧氏距离，9999 次置换）")
    print("=" * 78)
    F, R2, p = permanova(D, codes, rng=rng)
    print(f"全局(flat/mod/tv): pseudo-F = {F:.3f}, R^2 = {R2:.4f}, p = {p:.4f}")

    print("\n两两 PERMANOVA（Holm 校正）：")
    pair_rows = []
    for a, b in combinations(LABELS, 2):
        m = df.CallType.isin([a, b]).to_numpy()
        Dsub = pdist(Z[m].values, metric="euclidean")
        csub = df.CallType[m].map({a: 0, b: 1}).to_numpy()
        F2, R2_2, p2 = permanova(Dsub, csub, rng=rng)
        pair_rows.append((f"{a} vs {b}", F2, R2_2, p2))
        print(f"  {a:5s} vs {b:5s}: pseudo-F = {F2:7.3f}, R^2 = {R2_2:.4f}, p_raw = {p2:.4f}")
    pr = holm([r[3] for r in pair_rows])
    print("  p_holm:", "  ".join(f"{r[0]}={p:.4f}" for r, p in zip(pair_rows, pr)))

    print("\n" + "=" * 78)
    print("Mann-Whitney U 检验（每特征三对两两，18 项，Holm 校正；Cliff's δ）")
    print("=" * 78)
    print(f"{'feature':<9}{'pair':<14}{'U':>9}{'p_raw':>9}{'p_holm':>9}{'delta':>9}")
    rows = []
    for f in FEAT6:
        for a, b in combinations(LABELS, 2):
            x = df.loc[df.CallType.eq(a), f].dropna()
            y = df.loc[df.CallType.eq(b), f].dropna()
            u, pr = mannwhitneyu(x, y, alternative="two-sided")
            delta = 2 * u / (len(x) * len(y)) - 1
            rows.append((f, a, b, u, pr, delta))
    ps = holm([r[4] for r in rows])
    for (f, a, b, u, pr, delta), ph in zip(rows, ps):
        print(f"{f:<9}{a+'/'+b:<14}{u:>9.1f}{pr:>9.4f}{ph:>9.4f}{delta:>+9.3f}")

    # PulseRate：仅 mod vs tv（flat 无观测）
    print("\nPulseRate（仅 mod vs tv，flat 无观测）：")
    x = df.loc[df.CallType.eq("mod"), "PulseRate"].dropna()
    y = df.loc[df.CallType.eq("tv"), "PulseRate"].dropna()
    u, pr = mannwhitneyu(x, y, alternative="two-sided")
    delta = 2 * u / (len(x) * len(y)) - 1
    print(f"  mod/tv: U = {u:.1f}, p = {pr:.4f}, delta = {delta:+.3f}, n_mod={len(x)}, n_tv={len(y)}")

    # 汇总写入 CSV
    out = Path(r"G:\亚太数模\非参数检验\nonparametric_tests.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    mwu = pd.DataFrame([{"feature": f, "groups": f"{a}/{b}", "U": u, "p_raw": pr,
                         "p_holm": ph, "Cliff_delta": delta}
                        for (f, a, b, u, pr, delta), ph in zip(rows, ps)])
    mwu.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[ok] MWU 结果 -> {out}")


if __name__ == "__main__":
    main()
