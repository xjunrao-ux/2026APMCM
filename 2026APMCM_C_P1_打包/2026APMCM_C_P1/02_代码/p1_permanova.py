# -*- coding: utf-8 -*-
"""
P1 统计建模：
  (1) Kruskal-Wallis H  —— 单指标筛查（每个特征一次）
  (2) PERMANOVA        —— 多指标综合（标准化欧氏距离，9999 次置换）
  (3) Mann-Whitney U   —— 数据不全指标（F02 / PulseRate）的补充检验
配套：PERMDISP（离散度齐性）、成对 PERMANOVA、距离度量敏感性、PCoA 图。
纯 numpy 实现（无 scipy）。
"""
import math
import os
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"
OUT = r"X:\2026APMCM\figures"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
FEATS = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
TYPES = ["flat", "mod", "tv"]
y = np.array(d["CallType"].tolist(), dtype=object)   # pandas3 的 .values 是 StringArray，须转 numpy
rng = np.random.default_rng(20261126)

# ---------------------------------------------------------------- 基础工具
def rankdata(x):
    x = np.asarray(x, float)
    s = pd.Series(x).rank().values
    return s

def tie_term(x):
    _, cnt = np.unique(np.asarray(x, float), return_counts=True)
    return float(np.sum(cnt ** 3 - cnt))

def kruskal(groups):
    allv = np.concatenate(groups); N = len(allv); k = len(groups)
    R = rankdata(allv); idx = 0; S = 0.0
    for g in groups:
        n = len(g); S += n * R[idx:idx + n].mean() ** 2; idx += n
    H = 12.0 / (N * (N + 1)) * S - 3 * (N + 1)
    T = tie_term(allv)
    if T > 0:
        H /= (1 - T / (N ** 3 - N))
    df = k - 1
    if df == 2:
        p = math.exp(-H / 2)
    else:
        z = ((H / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return H, p, (H - k + 1) / (N - k)

def mwu(a, b, perms=0):
    """U, p(正态近似), Cliff's delta; perms>0 时改用置换 p（应对大量并列）"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b]); R = rankdata(allv)
    U = R[:n1].sum() - n1 * (n1 + 1) / 2
    N = n1 + n2
    if perms:
        obs = U
        cnt = 0; cnt2 = 0
        for _ in range(perms):
            idx = rng.permutation(N)
            Uu = rankdata(allv[idx])[:n1].sum() - n1 * (n1 + 1) / 2
            if abs(Uu - n1 * n2 / 2) >= abs(obs - n1 * n2 / 2):
                cnt += 1
        p = (cnt + 1) / (perms + 1)
    else:
        mu = n1 * n2 / 2
        T = tie_term(allv)
        sig = math.sqrt(n1 * n2 / 12 * ((N + 1) - T / (N * (N - 1))))
        z = (U - mu - 0.5) / sig if U > mu else (U - mu + 0.5) / sig
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U, p, 2 * U / (n1 * n2) - 1

def bh(p):
    p = np.asarray(p, float); n = len(p); o = p.argsort(); q = np.empty(n); prev = 1.0
    for r in range(n, 0, -1):
        i = o[r - 1]; prev = min(prev, p[i] * n / r); q[i] = prev
    return q

def dist_matrix(X, metric="euclid"):
    if metric == "euclid":
        sq = (X ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
        return np.sqrt(np.maximum(D2, 0))
    if metric == "manhattan":
        return np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    raise ValueError(metric)

# ---------------------------------------------------------------- PERMANOVA
def permanova(Dist, labels, perms=9999):
    D2 = Dist ** 2
    n = len(labels); uni = np.unique(labels); a = len(uni)
    Y = (labels[:, None] == uni[None, :]).astype(float)
    SST = D2.sum() / 2 / n

    def stat(lab):
        M = (lab[:, None] == uni[None, :]).astype(float)
        A = M.T @ D2 @ M
        nk = M.sum(0)
        SW = np.sum(np.diag(A) / 2 / nk)
        F = ((SST - SW) / (a - 1)) / (SW / (n - a))
        return F, (SST - SW) / SST

    F0, R2 = stat(labels)
    cnt = 0
    for _ in range(perms):
        if stat(rng.permutation(labels))[0] >= F0:
            cnt += 1
    return F0, R2, (cnt + 1) / (perms + 1)

def permdisp(Dist, labels, perms=9999):
    """PERMDISP：各组到组质心的平均距离是否相同（PCoA 坐标还原欧氏距离）"""
    D2 = Dist ** 2; n = len(labels)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    w = np.clip(w, 0, None)
    G = V @ np.diag(np.sqrt(w))
    uni = np.unique(labels)

    def stat(lab):
        r = np.array([np.linalg.norm(G[lab == u] - G[lab == u].mean(0), axis=1).mean() for u in uni])
        ng = np.array([(lab == u).sum() for u in uni], float)
        grand = (ng * r).sum() / ng.sum()
        return (ng * (r - grand) ** 2).sum(), r

    F0, r0 = stat(labels)
    cnt = sum(stat(rng.permutation(labels))[0] >= F0 for _ in range(perms))
    return F0, r0, (cnt + 1) / (perms + 1)

# ---------------------------------------------------------------- 1. KWH
print("=" * 86)
print("【一】Kruskal-Wallis H：单指标筛查（n=108，三组 flat44/mod40/tv24）")
print("=" * 86)
print(f"{'指标':10s}{'H':>9s}{'p':>10s}{'ε²':>9s}{'效应':>7s}   组中位数 flat / mod / tv")
kw_rows = []
for f in FEATS:
    gs = [d.loc[d.CallType == t, f].values for t in TYPES]
    H, p, eps = kruskal(gs)
    eff = "大" if eps >= .14 else ("中" if eps >= .06 else "小")
    med = [np.median(g) for g in gs]
    print(f"{f:10s}{H:9.3f}{p:10.4f}{eps:9.4f}{eff:>7s}   {med[0]:7.1f} /{med[1]:7.1f} /{med[2]:7.1f}")
    kw_rows.append((f, H, p, eps))
kw_q = bh([r[2] for r in kw_rows])
print("\nBH 校正后（6 次检验）:")
for (f, H, p, eps), q in zip(kw_rows, kw_q):
    print(f"  {f:10s} p={p:.4f}  q={q:.4f}  {'★显著' if q < .05 else '不显著'}")

# ---------------------------------------------------------------- 2. PERMANOVA
print("\n" + "=" * 86)
print("【二】PERMANOVA：多指标综合（6 个共享特征，z 标准化，欧氏距离，9999 次置换）")
print("=" * 86)
X = d[FEATS].values.astype(float)
Z = (X - X.mean(0)) / X.std(0, ddof=1)

# 共线性提示
C = np.corrcoef(Z.T)
print("特征相关矩阵（|r|>0.7 视为强共线）:")
print(pd.DataFrame(C, index=FEATS, columns=FEATS).round(2).to_string())

perm = 9999
F0, R2, p = permanova(dist_matrix(Z), y, perm)
print(f"\n整体: pseudo-F = {F0:.3f},  R² = {R2:.4f},  p = {p:.4f}  ({perm} 次置换)")
print(f"       → 组间差异解释了总变异的 {R2*100:.1f}%")

print("\n成对 PERMANOVA（3 对，9999 次置换，BH 校正）:")
rows = []
for i, a in enumerate(TYPES):
    for b in TYPES[i + 1:]:
        m = np.isin(y, [a, b])
        F, r2, pp = permanova(dist_matrix(Z[m]), y[m], perm)
        rows.append([f"{a}-{b}", F, r2, pp])
q = bh([r[3] for r in rows])
for r, qq in zip(rows, q):
    print(f"  {r[0]:10s} pseudo-F={r[1]:6.3f}  R²={r[2]:.4f}  p={r[3]:.4f}  q={qq:.4f}  {'★' if qq < .05 else ''}")

print("\nPERMDISP（多元离散度齐性）：")
Fd, res, pd_ = permdisp(dist_matrix(Z), y, 999)
print(f"  各组到质心平均距离: " + ", ".join(f"{t}={v:.2f}" for t, v in zip(TYPES, res)))
print(f"  置换检验 p = {pd_:.4f}  →{'离散度不同，PERMANOVA 结果可能受离散度影响' if pd_ < .05 else '离散度齐性成立，PERMANOVA 反映位置差异'}")

print("\n距离度量敏感性:")
for name, Dm in [("欧氏(标准化)", dist_matrix(Z)),
                 ("曼哈顿(标准化)", dist_matrix(Z, "manhattan"))]:
    F, r2, pp = permanova(Dm, y, 999)
    print(f"  {name:14s} pseudo-F={F:6.3f}  R²={r2:.4f}  p={pp:.4f}")

# 前 2 主成分上的 PERMANOVA（去共线）
w, V = np.linalg.eigh(C)
o = w.argsort()[::-1]; w, V = w[o], V[:, o]
pcs = Z @ V[:, :2]
ev = w / w.sum() * 100
print(f"  PCA: PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%, 累计={ev[:2].sum():.1f}%")
F, r2, pp = permanova(dist_matrix(pcs), y, 999)
print(f"  前2主成分     pseudo-F={F:6.3f}  R²={r2:.4f}  p={pp:.4f}")

# ---------------------------------------------------------------- 3. MWU 补充
print("\n" + "=" * 86)
print("【三】Mann-Whitney U：数据不全指标的补充检验")
print("=" * 86)
print("\n3.1 PulseRate 实测值（flat 无脉冲串，故只比 mod vs tv）")
a = d.loc[d.CallType == "mod", "PulseRate"].dropna().values
b = d.loc[d.CallType == "tv", "PulseRate"].dropna().values
U, p_, delta = mwu(a, b)
print(f"  mod(n={len(a)}, med={np.median(a):.1f}) vs tv(n={len(b)}, med={np.median(b):.1f})")
print(f"  U={U:.1f}, p={p_:.4f}, Cliff's δ={delta:.3f}  {'显著' if p_ < .05 else '不显著'}")

print("\n3.2 F02：仅 tv 有值 → 编码为『缺失=0』后做 MWU（含置换 p，因大量并列）")
d["F02star"] = d["F02"].fillna(0.0)
for a_, b_ in [("flat", "tv"), ("mod", "tv"), ("flat", "mod")]:
    va = d.loc[d.CallType == a_, "F02star"].values
    vb = d.loc[d.CallType == b_, "F02star"].values
    U, p_, delta = mwu(va, vb, perms=2000)
    print(f"  {a_:5s}(n={len(va):2d}) vs {b_:5s}(n={len(vb):2d}): U={U:7.1f}, p={p_:.4f}, Cliff's δ={delta:+.3f}")

print("\n3.3 PulseRate 编码为『缺失=0』后做 MWU")
d["PRstar"] = d["PulseRate"].fillna(0.0)
for a_, b_ in [("flat", "tv"), ("flat", "mod"), ("mod", "tv")]:
    va = d.loc[d.CallType == a_, "PRstar"].values
    vb = d.loc[d.CallType == b_, "PRstar"].values
    U, p_, delta = mwu(va, vb, perms=2000)
    print(f"  {a_:5s} vs {b_:5s}: U={U:7.1f}, p={p_:.4f}, Cliff's δ={delta:+.3f}")

# ---------------------------------------------------------------- 4. 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Dm = dist_matrix(Z)
n = len(y)
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (Dm ** 2) @ J
wq, Vq = np.linalg.eigh(B)
o = wq.argsort()[::-1]; wq, Vq = wq[o], Vq[:, o]
coords = Vq[:, :2] * np.sqrt(np.clip(wq[:2], 0, None))
varx = wq[:2] / wq[wq > 0].sum() * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
colors = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}
for t in TYPES:
    m = y == t
    axes[0].scatter(coords[m, 0], coords[m, 1], s=42, alpha=.75,
                    c=colors[t], label=f"{t} (n={m.sum()})", edgecolor="w", lw=.6)
axes[0].set_xlabel(f"PCoA 1 ({varx[0]:.1f}%)"); axes[0].set_ylabel(f"PCoA 2 ({varx[1]:.1f}%)")
axes[0].set_title(f"Multivariate space (PERMANOVA R²={R2:.3f}, p={p:.3f})")
axes[0].legend(frameon=False, fontsize=9); axes[0].axhline(0, lw=.4, c="grey"); axes[0].axvline(0, lw=.4, c="grey")

for ax, f in zip(axes[1:], ["Fpeak", "F01"]):
    data = [d.loc[d.CallType == t, f].values for t in TYPES]
    bp = ax.boxplot(data, patch_artist=True, widths=.55, showfliers=True,
                    medianprops=dict(color="black", lw=1.4))
    for patch, t in zip(bp["boxes"], TYPES):
        patch.set_facecolor(colors[t]); patch.set_alpha(.65)
    ax.set_xticklabels(TYPES); ax.set_ylabel(f + (" (Hz)" if f != "Duration" else " (s)"))
    ax.set_title(f"{f}: Kruskal-Wallis p={kw_rows[FEATS.index(f)][2]:.4f}")

plt.tight_layout()
png = os.path.join(OUT, "p1_kw_permanova.png")
plt.savefig(png, dpi=200, bbox_inches="tight")
print(f"\n[图已保存] {png}")

print("\n" + "=" * 86)
print("【四】三方法结论对照")
print("=" * 86)
print(f"KWH   ：显著指标 = " + ", ".join(f"{f}(q={qq:.3f})" for (f, H, p_, eps), qq in zip(kw_rows, kw_q) if qq < .05))
print(f"PERMANOVA：整体伪F={F0:.2f}, R²={R2:.4f}, p={p:.4f}")
print(f"MWU   ：F02 与 PulseRate 的结构差异（见 3.1-3.3）")
