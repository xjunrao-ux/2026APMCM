# -*- coding: utf-8 -*-
"""
生成三张结果分析示意图：
  fig1 显著性的来源：置换零分布 / 原始p与BH-q对照 / 效应量-显著性平面
  fig2 数据怎么读：Fpeak 双峰 / 峰值位置比例 / PCoA+95%椭圆 / PERMDISP
  fig3 效应量森林图：18 组两两比较的 Cliff's δ 及 bootstrap CI
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ---------------- 中文字体
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

D = r"X:\2026APMCM\题目与要求\date"
OUT = r"X:\2026APMCM\figures"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
FEATS = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
TYPES = ["flat", "mod", "tv"]
COL = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}
y = np.array(d["CallType"].tolist(), dtype=object)
rng = np.random.default_rng(7)


def fast_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def tie_term(x):
    _, c = np.unique(np.asarray(x, float), return_counts=True)
    return float(np.sum(c ** 3 - c))


def kruskal(groups):
    allv = np.concatenate(groups); N = len(allv); k = len(groups)
    R = fast_rank(allv); idx = 0; S = 0.0
    for g in groups:
        n = len(g); S += n * R[idx:idx + n].mean() ** 2; idx += n
    H = 12.0 / (N * (N + 1)) * S - 3 * (N + 1)
    T = tie_term(allv)
    if T > 0:
        H /= (1 - T / (N ** 3 - N))
    p = math.exp(-H / 2)
    return H, p, (H - k + 1) / (N - k)


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b]); R = fast_rank(allv)
    U = R[:n1].sum() - n1 * (n1 + 1) / 2
    N = n1 + n2; mu = n1 * n2 / 2
    T = tie_term(allv)
    sig = math.sqrt(n1 * n2 / 12 * ((N + 1) - T / (N * (N - 1))))
    if sig == 0:
        return U, 1.0, 2 * U / (n1 * n2) - 1
    z = (U - mu - 0.5) / sig if U > mu else (U - mu + 0.5) / sig
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U, p, 2 * U / (n1 * n2) - 1


def bh(p):
    p = np.asarray(p, float); n = len(p); o = p.argsort(); q = np.empty(n); prev = 1.0
    for r in range(n, 0, -1):
        i = o[r - 1]; prev = min(prev, p[i] * n / r); q[i] = prev
    return q


X = d[FEATS].values.astype(float)
Z = (X - X.mean(0)) / X.std(0, ddof=1)


def dist_matrix(A):
    sq = (A ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0))


Dt = dist_matrix(Z)
D2 = Dt ** 2
n = len(y)
uni = np.array(TYPES, dtype=object)


def pstat(lab):
    M = (lab[:, None] == uni[None, :]).astype(float)
    A = M.T @ D2 @ M
    nk = M.sum(0)
    SST = D2.sum() / 2 / n
    SW = np.sum(np.diag(A) / 2 / nk)
    return ((SST - SW) / 2) / (SW / (n - 3)), (SST - SW) / SST


F0, R2 = pstat(y)
nullF = np.array([pstat(rng.permutation(y))[0] for _ in range(2000)])
pval = (np.sum(nullF >= F0) + 1) / (len(nullF) + 1)

# ---- 18 组 MWU
rows = []
for f in FEATS:
    for a, b in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
        U, p, delta = mwu(d.loc[d.CallType == a, f].values, d.loc[d.CallType == b, f].values)
        rows.append(dict(feature=f, pair=f"{a}–{b}", U=U, p=p, delta=delta))
tab = pd.DataFrame(rows)
tab["q"] = bh(tab.p.values)
tab = tab.sort_values("p").reset_index(drop=True)

kw = [(f, *kruskal([d.loc[d.CallType == t, f].values for t in TYPES])) for f in FEATS]

# ======================================================= 图 1
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

a0 = ax[0]
a0.hist(nullF, bins=45, color="#9aa7b8", edgecolor="w", lw=.4, label="置换零分布（2000 次）")
a0.axvline(F0, color="#C44E52", lw=2.2, label=f"观测 pseudo-F = {F0:.2f}")
cut = np.percentile(nullF, 95)
a0.axvline(cut, color="k", ls=":", lw=1.2, label="零分布 95% 分位")
a0.set_xlabel("pseudo-F"); a0.set_ylabel("频数")
a0.set_title(f"(a) 置换检验怎么给 p 值\np = 比观测更极端的比例 = {pval:.4f}", fontsize=11)
a0.legend(fontsize=8, frameon=False, loc="upper right")
a0.annotate("观测值落在零分布右尾\n→ 差异不能由随机解释",
            xy=(F0, 28), xytext=(F0 + 0.12, 78),
            fontsize=8.5, color="#C44E52",
            arrowprops=dict(arrowstyle="->", lw=1, color="#C44E52"))

a1 = ax[1]
xs = np.arange(len(tab))
lp = -np.log10(np.clip(tab.p.values, 1e-5, None))
lq = -np.log10(np.clip(tab.q.values, 1e-5, None))
a1.vlines(xs, lq, lp, color="#cccccc", lw=1.2, zorder=1)
a1.scatter(xs, lp, s=34, c="#7f8c8d", zorder=3, label="原始 p")
a1.scatter(xs, lq, s=34, c=np.where(tab.q < .05, "#C44E52", "#bdc3c7"), zorder=4,
           label="BH 校正后 q")
a1.axhline(-math.log10(.05), color="k", ls="--", lw=1)
a1.text(len(tab) - .4, -math.log10(.05) + .06, "α = 0.05", ha="right", fontsize=8.5)
a1.set_xticks(xs); a1.set_xticklabels([f"{r.feature}\n{r.pair}" for r in tab.itertuples()],
                                     rotation=90, fontsize=6.5)
a1.set_ylabel("−log10(p) / −log10(q)")
a1.set_title("(b) 多重比较校正前后的对照\n18 次检验，只有 1 次真正过线", fontsize=11)
a1.legend(fontsize=8, frameon=False, loc="upper right")

a2 = ax[2]
xs2 = [k[3] for k in kw]; ys2 = [-math.log10(k[2]) for k in kw]
qk = bh([k[2] for k in kw])
for (f, H, p_, eps), qq, xx, yy in zip(kw, qk, xs2, ys2):
    a2.scatter(xx, yy, s=110, c="#C44E52" if qq < .05 else "#9aa7b8",
               edgecolor="k", lw=.6, zorder=3)
    a2.annotate(f, (xx, yy), textcoords="offset points", xytext=(6, 5), fontsize=9)
a2.axhline(-math.log10(.05), color="k", ls="--", lw=1)
a2.axvline(.06, color="#4C72B0", ls=":", lw=1.3)
a2.text(.061, .05, "ε²=0.06\n中效应门槛", fontsize=8, color="#4C72B0")
a2.text(.0, -math.log10(.05) + .04, "p = 0.05", fontsize=8.5)
a2.set_xlabel("效应量 ε²"); a2.set_ylabel("−log10(p)")
a2.set_title("(c) 显著 ≠ 重要\n纵轴是“有没有”，横轴是“有多大”", fontsize=11)
a2.set_xlim(-.03, .12); a2.set_ylim(0, 3.0)

plt.tight_layout()
p1 = os.path.join(OUT, "fig1_显著性的来源.png")
plt.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig)
print("[保存]", p1)

# ======================================================= 图 2
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

b0 = ax[0, 0]
bins = np.linspace(150, 1350, 26)
for t in TYPES:
    b0.hist(d.loc[d.CallType == t, "Fpeak"], bins=bins, alpha=.5, color=COL[t], label=t)
b0.axvline(600, color="k", ls="--", lw=1.2)
b0.text(615, b0.get_ylim()[1] * .82, "600 Hz\n← 基频区 | 谐波区 →", fontsize=8.5)
b0.set_xlabel("Fpeak (Hz)"); b0.set_ylabel("呼叫数")
b0.set_title("(a) Fpeak 是双峰，不是单峰\n三类差别在“两个峰的比例”，不在均值", fontsize=11)
b0.legend(frameon=False, fontsize=9)

b1 = ax[0, 1]
ratio = d.Fpeak / d.F01
d["peakpos"] = np.where(ratio < 2, "峰值在基频", "峰值在第3谐波")
ct = pd.crosstab(d.CallType, d.peakpos, normalize="index").loc[TYPES]
bottom = np.zeros(3)
for col, cc in [("峰值在基频", "#C44E52"), ("峰值在第3谐波", "#9aa7b8")]:
    b1.bar(TYPES, ct[col], bottom=bottom, color=cc, label=col, width=.6, edgecolor="w")
    for i, v in enumerate(ct[col]):
        if v > .08:
            b1.text(i, bottom[i] + v / 2, f"{v*100:.0f}%", ha="center", va="center",
                    color="w", fontsize=10, fontweight="bold")
    bottom += ct[col].values
b1.set_ylabel("比例"); b1.set_ylim(0, 1)
b1.set_title("(b) 同一个 Fpeak，两类呼叫的构成不同\nflat 仅 11%，mod 38%、tv 33%", fontsize=11)
b1.legend(frameon=False, fontsize=9, loc="upper center",
          bbox_to_anchor=(.5, -.08), ncol=2)

b2 = ax[1, 0]
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ D2 @ J
wq, Vq = np.linalg.eigh(B)
o = wq.argsort()[::-1]; wq, Vq = wq[o], Vq[:, o]
C = Vq[:, :2] * np.sqrt(np.clip(wq[:2], 0, None))
ev = wq[:2] / wq[wq > 0].sum() * 100
for t in TYPES:
    m = y == t
    b2.scatter(C[m, 0], C[m, 1], s=30, alpha=.55, c=COL[t], label=t, edgecolor="w", lw=.4)
    mu = C[m].mean(0)
    cov = np.cov(C[m].T)
    vals, vecs = np.linalg.eigh(cov)
    od = vals.argsort()[::-1]; vals, vecs = vals[od], vecs[:, od]
    th = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    e = mpatches.Ellipse(mu, 2 * 2.4477 * math.sqrt(vals[0]), 2 * 2.4477 * math.sqrt(vals[1]),
                         angle=th, fill=False, lw=1.8, ls="--", color=COL[t])
    b2.add_patch(e)
    b2.scatter(*mu, marker="X", s=180, c=COL[t], edgecolor="k", lw=1, zorder=5)
b2.set_xlabel(f"PCoA 1（{ev[0]:.1f}%）"); b2.set_ylabel(f"PCoA 2（{ev[1]:.1f}%）")
b2.set_title(f"(c) 三个椭圆大面积重叠：R²={R2:.3f}\n× 为组质心，重叠越多越难分", fontsize=11)
b2.legend(frameon=False, fontsize=9); b2.axhline(0, lw=.4, c="grey"); b2.axvline(0, lw=.4, c="grey")

b3 = ax[1, 1]
Gfull = Vq @ np.diag(np.sqrt(np.clip(wq, 0, None)))
dcen = [np.linalg.norm(Gfull[y == t] - Gfull[y == t].mean(0), axis=1) for t in TYPES]
bp = b3.boxplot(dcen, patch_artist=True, widths=.55,
                medianprops=dict(color="k", lw=1.4))
for patch, t in zip(bp["boxes"], TYPES):
    patch.set_facecolor(COL[t]); patch.set_alpha(.65)
b3.set_xticklabels(TYPES); b3.set_ylabel("到组质心的距离")
b3.set_title("(d) 离散度齐性检验 PERMDISP：p = 0.159\n三组“散开程度”相当 → 差异属位置差异", fontsize=11)
for i, t in enumerate(TYPES):
    b3.text(i + 1, np.asarray(dcen[i]).max() * 1.02, f"均值 {np.asarray(dcen[i]).mean():.2f}",
            ha="center", fontsize=8.5)

plt.tight_layout()
p2 = os.path.join(OUT, "fig2_数据怎么读.png")
plt.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)
print("[保存]", p2)

# ======================================================= 图 3 森林图
boot = 1000
recs = []
for r in tab.itertuples():
    fa, fb = r.pair.split("–")
    for fa, fb, src, dst in [(fa, fb, fa, fb)]:
        pass
for f in FEATS:
    for a, b in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
        va = d.loc[d.CallType == a, f].values
        vb = d.loc[d.CallType == b, f].values
        _, _, delta = mwu(va, vb)
        ds = []
        for _ in range(boot):
            sa = rng.choice(va, len(va), replace=True)
            sb = rng.choice(vb, len(vb), replace=True)
            ds.append(mwu(sa, sb)[2])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        pr = tab[(tab.feature == f) & (tab.pair == f"{a}–{b}")].iloc[0]
        recs.append(dict(label=f"{f}   {a} vs {b}", delta=delta, lo=lo, hi=hi,
                         q=pr.q, pair=f"{a}–{b}"))
F = pd.DataFrame(recs)
F["absd"] = F.delta.abs()
F = F.sort_values("absd").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.6, 7.4))
ys = np.arange(len(F))
for i, r in F.iterrows():
    sig = r.q < .05
    cc = "#C44E52" if sig else "#7f8c8d"
    ax.plot([r.lo, r.hi], [i, i], color=cc, lw=2, alpha=.85, solid_capstyle="round")
    ax.scatter(r.delta, i, s=85, c=cc, edgecolor="w", lw=.8, zorder=4)
ax.axvline(0, color="k", lw=1.1)
for x, lab, cc in [(.147, "小", "#bdc3c7"), (.33, "中", "#95a5a6"), (.474, "大", "#7f8c8d")]:
    for s in (-1, 1):
        ax.axvline(s * x, color=cc, ls=":", lw=.8, zorder=0)
ax.text(.02, .985, "效应量标尺（点线，左右对称）：小 0.147 · 中 0.33 · 大 0.474",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#7f8c8d")
ax.set_yticks(ys); ax.set_yticklabels(F.label, fontsize=9)
ax.set_xlabel("Cliff's δ（>0 表示前者的值更大）", fontsize=10)
ax.set_title("三类呼叫两两比较的效应量及 95% bootstrap 置信区间\n红色 = BH 校正后仍显著；区间跨 0 即无可判定的差异", fontsize=11.5)
ax.set_xlim(-1.0, 1.0)
ax.legend(handles=[mpatches.Patch(color="#C44E52", label="q < 0.05"),
                   mpatches.Patch(color="#7f8c8d", label="q ≥ 0.05")],
          frameon=False, loc="lower right", fontsize=9)

plt.tight_layout()
p3 = os.path.join(OUT, "fig3_效应量森林图.png")
plt.savefig(p3, dpi=200, bbox_inches="tight"); plt.close(fig)
print("[保存]", p3)

print("\n--- 关键数值 ---")
print(f"PERMANOVA: F={F0:.3f}, R²={R2:.4f}, p={pval:.4f}, 零分布均值={nullF.mean():.3f}")
print(f"零分布 95% 分位={cut:.3f}；观测值超出零点布 {np.sum(nullF>=F0)}/{len(nullF)}")
print(f"过 BH 的 MWU 对数: {(tab.q<.05).sum()}/{len(tab)} -> {tab[tab.q<.05][['feature','pair']].values.tolist()}")
print("最大 |δ|:", F.iloc[-1].label, round(F.iloc[-1].delta, 3))
