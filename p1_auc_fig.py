# -*- coding: utf-8 -*-
"""AUC 对比：配对 bootstrap 差异检验 + 出图"""
import os
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
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
d["hasPR"] = d.PulseRate.notna().astype(int)
d["hasF02"] = d.F02.notna().astype(int)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
rng = np.random.default_rng(9)

dd = d[d.CallType != "tv"].reset_index(drop=True)
y = (dd.CallType == "mod").astype(int).values
prior = y[dd.hasPR.values == 0].mean()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit_fit(X, yy, lr=.5, iters=4000, lam=1e-2):
    n, p = X.shape
    w = np.zeros(p + 1); Xb = np.column_stack([np.ones(n), X])
    for _ in range(iters):
        w -= lr * (Xb.T @ (sigmoid(Xb @ w) - yy) / n + lam * np.r_[0, w[1:]])
    return w


def loo_prob(X, yy):
    out = np.zeros(len(yy))
    for i in range(len(yy)):
        tr = np.ones(len(yy), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, yy[tr])
        out[i] = sigmoid(np.r_[1, (X[i] - mu) / sd] @ w)
    return out


def auc(score, yy):
    s = pd.Series(score).rank().values
    n1 = yy.sum(); n0 = len(yy) - n1
    return (s[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


feats6 = dd[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]].values.astype(float)
S = {
    "你的三层方法": np.where(dd.hasPR.values == 1, 1.0, prior),
    "6 声学量+脉冲率": loo_prob(np.column_stack([feats6, dd.hasPR.values]), y),
    "Fpeak+脉冲率": loo_prob(np.column_stack([dd.Fpeak.values, dd.hasPR.values]), y),
    "6 个声学量": loo_prob(feats6, y),
    "Fpeak（方向修正）": -dd.Fpeak.values,
    "峰值位置": dd.peakpos.values.astype(float),
    "F01（方向修正）": dd.F01.values,
}
B = 2000
idx1 = np.where(y == 1)[0]; idx0 = np.where(y == 0)[0]
res = {}
boots = {}
for k, v in S.items():
    a = auc(v, y)
    bs = []
    for _ in range(B):
        i1 = rng.choice(idx1, len(idx1)); i0 = rng.choice(idx0, len(idx0))
        ii = np.r_[i1, i0]; tt = np.r_[np.ones(len(i1)), np.zeros(len(i0))]
        bs.append(auc(v[ii], tt))
    bs = np.array(bs)
    boots[k] = bs
    res[k] = (a, np.percentile(bs, 2.5), np.percentile(bs, 97.5))

print("=" * 78)
print("AUC 与配对 bootstrap 差异检验（B=2000）")
print("=" * 78)
print(f"{'方案':22s}{'AUC':>8s}{'95%CI':>18s}")
for k, (a, lo, hi) in sorted(res.items(), key=lambda z: -z[1][0]):
    print(f"{k:22s}{a:8.4f}   [{lo:.3f}, {hi:.3f}]")

print("\n与你方法的配对差异（ΔAUC 及 95%CI，跨 0 即无差别）：")
ref = boots["你的三层方法"]
for k in ["6 声学量+脉冲率", "Fpeak+脉冲率", "6 个声学量", "Fpeak（方向修正）"]:
    dif = ref - boots[k]
    lo, hi = np.percentile(dif, [2.5, 97.5])
    p = 2 * min((dif <= 0).mean(), (dif >= 0).mean())
    print(f"  vs {k:20s} Δ={res['你的三层方法'][0]-res[k][0]:+.4f}  "
          f"CI=[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  "
          f"{'显著' if lo*hi > 0 else '不显著'}")

# ---------------- 图
items = sorted(res.items(), key=lambda z: z[1][0])
labels = [k for k, _ in items]
vals = np.array([v[0] for _, v in items])
los = np.array([v[1] for _, v in items])
his = np.array([v[2] for _, v in items])
colors = []
for k in labels:
    if k == "你的三层方法":
        colors.append("#C44E52")
    elif "脉冲率" in k:
        colors.append("#55A868")
    elif "声学量" in k:
        colors.append("#4C72B0")
    else:
        colors.append("#9aa7b8")

fig, ax = plt.subplots(figsize=(9.2, 5.2))
yy = np.arange(len(labels))
ax.barh(yy, vals, color=colors, alpha=.88, edgecolor="w", height=.62)
ax.errorbar(vals, yy, xerr=[vals - los, his - vals], fmt="none",
            ecolor="k", elinewidth=1.2, capsize=4)
for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
    ax.text(min(hi + .015, .995), i, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(.5, color="k", ls="--", lw=1.1)
ax.text(.505, -.7, "随机猜测 AUC=0.5", fontsize=8.5)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(.35, 1.02); ax.set_xlabel("AUC（95% bootstrap 置信区间）", fontsize=10)
ax.set_title("flat vs mod（84 例）判别力对比\n红色=你的方法，绿色=含脉冲率，蓝色=多特征回归，灰=单特征", fontsize=11)
ax.grid(axis="x", alpha=.25)
plt.tight_layout()
p = r"X:\2026APMCM\figures\fig6_AUC对比.png"
plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"\n[保存] {p}")
