# -*- coding: utf-8 -*-
"""P1 非参数检验：Kruskal-Wallis + Mann-Whitney U + 效应量 + BH 校正。
纯 numpy/math 实现（秩检验不需要分布假设，但 p 值用卡方/正态近似）。"""
import math
import pandas as pd
import numpy as np

D = r"X:\2026APMCM\题目与要求\date"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
FEATS = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]


def rankdata(x):
    """平均秩（处理并列）"""
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # 并列取平均秩
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    return ranks


def tie_term(x):
    _, cnt = np.unique(x, return_counts=True)
    return float(np.sum(cnt ** 3 - cnt))


def kruskal(groups):
    """返回 H, p(df=k-1), epsilon^2, 是否需并列校正"""
    allv = np.concatenate(groups)
    N = len(allv)
    R = rankdata(allv)
    k = len(groups)
    idx = 0
    H = 0.0
    for g in groups:
        n = len(g)
        Rbar = R[idx:idx + n].mean()
        H += n * Rbar ** 2
        idx += n
    H = 12.0 / (N * (N + 1)) * H - 3 * (N + 1)
    # 并列校正
    T = tie_term(allv)
    if T > 0:
        H /= (1 - T / (N ** 3 - N))
    # df=2 时卡方 CDF 有闭式: p = exp(-H/2)；一般情形用 Wilson-Hilferty 近似
    df = k - 1
    if df == 2:
        p = math.exp(-H / 2)
    else:
        z = ((H / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    eps2 = (H - k + 1) / (N - k)          # 效应量 epsilon^2
    return H, p, eps2


def mannwhitney(a, b):
    """返回 U, p(正态近似+并列校正), Cliff's delta, rank-biserial r"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    R = rankdata(allv)
    R1 = R[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    N = n1 + n2
    T = tie_term(allv)
    sigma = math.sqrt(n1 * n2 / 12 * ((N + 1) - T / (N * (N - 1))))
    z = (U1 - mu - 0.5) / sigma if U1 > mu else (U1 - mu + 0.5) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    delta = 2 * U1 / (n1 * n2) - 1          # Cliff's delta
    r_rb = 1 - 2 * U1 / (n1 * n2)           # rank-biserial (方向)
    return U1, p, delta, r_rb


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = p.argsort()
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


print("=" * 78)
print("一、三组总体差异：Kruskal-Wallis（每个特征一次）")
print("=" * 78)
print(f"{'特征':10s} {'H':>8s} {'p':>10s} {'epsilon^2':>10s}  判读")
for f in FEATS:
    gs = [d.loc[d.CallType == t, f].values for t in ["flat", "mod", "tv"]]
    H, p, eps = kruskal(gs)
    verdict = "显著" if p < 0.05 else "不显著"
    strength = "强" if eps >= .14 else ("中" if eps >= .06 else "弱")
    print(f"{f:10s} {H:8.3f} {p:10.4f} {eps:10.4f}  {verdict} (效应{strength})")

print()
print("=" * 78)
print("二、两两事后：Mann-Whitney U（3 对 × 6 特征 = 18 次，BH 校正）")
print("=" * 78)
rows = []
for f in FEATS:
    for a, b in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
        U, p, delta, r = mannwhitney(d.loc[d.CallType == a, f].values,
                                     d.loc[d.CallType == b, f].values)
        rows.append(dict(feature=f, pair=f"{a}-{b}", U=U, p=p, cliff=delta))
tab = pd.DataFrame(rows)
tab["q_BH"] = bh(tab.p.values)
tab["sig"] = np.where(tab.q_BH < .05, "★", "")
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print()
print("=" * 78)
print("三、结构特征：F02 / PulseRate 的『有无』——卡方不适用，须精确检验")
print("=" * 78)
for col, lab in [("F02", "F02(第二基频)"), ("PulseRate", "PulseRate(脉冲率)")]:
    has = d[col].notna()
    ct = pd.crosstab(d.CallType, has)
    print(f"\n[{lab}] 列联表 (False=无, True=有):")
    print(ct.to_string())
    # 置换检验：打乱标签 5000 次，看组间"有"的比例差异
    rng = np.random.default_rng(1)
    obs = d.groupby("CallType")[col].apply(lambda s: s.notna().mean())
    obs_stat = obs.max() - obs.min()
    cnt = 0
    y = has.values
    for _ in range(5000):
        yp = rng.permutation(y)
        g = pd.Series(yp).groupby(d.CallType.values).mean()
        if (g.max() - g.min()) >= obs_stat:
            cnt += 1
    print(f"  各组『有』的比例: {obs.round(3).to_dict()}")
    print(f"  置换检验 p = {cnt/5000:.4f}  （差异统计量={obs_stat:.3f}）")
