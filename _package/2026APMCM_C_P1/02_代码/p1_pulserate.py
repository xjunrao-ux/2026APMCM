# -*- coding: utf-8 -*-
"""PulseRate 是否能作为显著差异指标：描述统计 + MWU + 检验力(MDE) + 图"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

D = r"X:\2026APMCM\题目与要求\date"
OUT = r"X:\2026APMCM\figures"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
TYPES = ["flat", "mod", "tv"]
COL = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}
rng = np.random.default_rng(11)


def fast_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort"); xs = x[order]
    r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def tie_term(x):
    _, c = np.unique(np.asarray(x, float), return_counts=True)
    return float(np.sum(c ** 3 - c))


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b]); R = fast_rank(allv)
    U = R[:n1].sum() - n1 * (n1 + 1) / 2
    N = n1 + n2; mu = n1 * n2 / 2
    sig = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie_term(allv) / (N * (N - 1))))
    z = (U - mu - .5) / sig if U > mu else (U - mu + .5) / sig
    p = 2 * (1 - .5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U, p, 2 * U / (n1 * n2) - 1


print("=" * 74)
print("一、PulseRate 的可得性与分布")
print("=" * 74)
for t in TYPES:
    s = d.loc[d.CallType == t, "PulseRate"].dropna()
    if len(s) == 0:
        print(f"  {t:5s}  n=0/{44 if t=='flat' else '?'}  —— 完全无数据（flat 呼叫无脉冲串）")
    else:
        tot = (d.CallType == t).sum()
        print(f"  {t:5s}  n={len(s):2d}/{tot}  中位数={s.median():5.1f}  "
              f"四分位=[{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]  "
              f"范围=[{s.min():.1f}, {s.max():.1f}]  CV={s.std()/s.mean():.2f}")

a = d.loc[d.CallType == "mod", "PulseRate"].dropna().values
b = d.loc[d.CallType == "tv", "PulseRate"].dropna().values
U, p, delta = mwu(a, b)
print("\n二、MWU：mod vs tv（唯一有数据可比的一对）")
print(f"  U={U:.1f}  p={p:.4f}  Cliff's δ={delta:+.3f}  AUC={(delta+1)/2:.3f}")
print(f"  → {'显著' if p < .05 else '不显著'}；效应量{'小' if abs(delta) < .147 else ('中' if abs(delta) < .33 else '大')}")

print("\n三、如果把『无脉冲串』编码为 0（结构指标）")
d["PRstar"] = d["PulseRate"].fillna(0.0)
for x, y_ in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
    va = d.loc[d.CallType == x, "PRstar"].values
    vb = d.loc[d.CallType == y_, "PRstar"].values
    U2, p2, d2 = mwu(va, vb)
    print(f"  {x:5s} vs {y_:5s}: p={p2:.4f}  δ={d2:+.3f}")

# ---------------- 检验力 / 最小可检出差异
print("\n" + "=" * 74)
print("四、检验力分析：n=(34,20) 时能检出多大的差异？")
print("=" * 74)
pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
print(f"  合并标准差 SD = {pooled_sd:.2f} 次/秒")
n1, n2 = len(a), len(b)


def power(dshift, reps=3000):
    cnt = 0
    for _ in range(reps):
        x = rng.normal(0, pooled_sd, n1)
        yv = rng.normal(dshift, pooled_sd, n2)
        if mwu(x, yv)[1] < .05:
            cnt += 1
    return cnt / reps


shifts = np.array([0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.2]) * pooled_sd
pw = np.array([power(s) for s in shifts])
print(f"  {'真实差异(Hz)':>12s}{'以SD计':>10s}{'检验力':>9s}")
for s, v in zip(shifts, pw):
    print(f"  {s:12.1f}{s/pooled_sd:10.2f}{v:9.2f}")
idx = np.argmax(pw >= .8)
mde = shifts[idx]
print(f"\n  80% 检验力所需的最小真实差异 ≈ {mde:.1f} 次/秒（约 {mde/pooled_sd:.2f} SD）")
print(f"  实测差异 = {abs(b.mean()-a.mean()):.1f} 次/秒 → 远低于该门槛")
print(f"  结论：n=34/20 只够检出『中到大』效应；本次不显著不能证明『无差异』")

# ---------------- 图
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
a0 = ax[0]
for i, t in enumerate(TYPES):
    s = d.loc[d.CallType == t, "PulseRate"].dropna().values
    if len(s) == 0:
        a0.text(i + .30, 46, "无脉冲串\n(n=0)", ha="center", va="center", fontsize=8.5,
                color=COL[t], bbox=dict(boxstyle="round,pad=.28", fc="white", ec=COL[t], ls="--", lw=1))
        continue
    a0.scatter(np.full(len(s), i) + rng.uniform(-.12, .12, len(s)), s,
               s=36, alpha=.75, c=COL[t], edgecolor="w", lw=.5)
    a0.plot([i - .25, i + .25], [np.median(s)] * 2, color="k", lw=2, zorder=5)
    a0.text(i, np.median(s) + 2.5, f"中位 {np.median(s):.1f}", ha="center", fontsize=8.5)
a0.set_xticks(range(3)); a0.set_xticklabels(TYPES)
a0.set_ylabel("PulseRate（次/秒）"); a0.set_ylim(0, 55)
a0.set_title("(a) 脉冲率：flat 完全无数据\nmod vs tv  p=%.3f, δ=%+.2f（不显著）" % (p, delta), fontsize=11)

a1 = ax[1]
a1.plot(shifts, pw, "o-", color="#4C72B0", lw=2, ms=5)
a1.axhline(.8, color="#C44E52", ls="--", lw=1.2)
a1.text(shifts[-1], .82, "80% 检验力", ha="right", fontsize=9, color="#C44E52")
a1.axvline(mde, color="#C44E52", ls=":", lw=1.2)
a1.text(mde + .8, .35, f"最小可检出\n≈{mde:.1f} 次/秒", fontsize=9, color="#C44E52")
a1.axvline(abs(b.mean() - a.mean()), color="#7f8c8d", ls=":", lw=1.2)
a1.text(abs(b.mean() - a.mean()) + .8, .55, f"实测差异\n≈{abs(b.mean()-a.mean()):.1f}",
        fontsize=9, color="#7f8c8d")
a1.set_xlabel("真实差异（次/秒）"); a1.set_ylabel("检验力（检出概率）")
a1.set_ylim(0, 1.02)
a1.set_title("(b) n=34/20 的检验力曲线\n只能检出中到大效应", fontsize=11)
a1.grid(alpha=.25)

plt.tight_layout()
fp = os.path.join(OUT, "fig4_脉冲率检验力.png")
plt.savefig(fp, dpi=200, bbox_inches="tight"); plt.close(fig)
print("\n[保存]", fp)
