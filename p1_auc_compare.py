# -*- coding: utf-8 -*-
"""你的三层方法 vs 其他方法的 AUC 对比（84 条 flat vs mod；以及三分类 macro-AUC）"""
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
d["hasPR"] = d.PulseRate.notna().astype(int)
d["hasF02"] = d.F02.notna().astype(int)
d["PR0"] = d.PulseRate.fillna(0.0)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
d["BW"] = d.Q75 - d.Q25
rng = np.random.default_rng(5)

sub = (d.CallType != "tv").values
dd = d[sub].reset_index(drop=True)
y = (dd.CallType == "mod").astype(int).values          # 1 = mod, 0 = flat
N = len(y)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit_fit(X, yy, lr=.5, iters=4000, lam=1e-2):
    n, p = X.shape
    w = np.zeros(p + 1)
    Xb = np.column_stack([np.ones(n), X])
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
    """Mann-Whitney AUC，并列按平均秩（0.5 计）"""
    s = pd.Series(score).rank().values
    n1 = yy.sum(); n0 = len(yy) - n1
    return (s[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, yy, B=2000):
    vals = []
    idx1 = np.where(yy == 1)[0]; idx0 = np.where(yy == 0)[0]
    for _ in range(B):
        i1 = rng.choice(idx1, len(idx1), replace=True)
        i0 = rng.choice(idx0, len(idx0), replace=True)
        s = np.r_[score[i1], score[i0]]; t = np.r_[np.ones(len(i1)), np.zeros(len(i0))]
        vals.append(auc(s, t))
    return np.percentile(vals, [2.5, 97.5])


print("=" * 92)
print("一、84 条（flat 44 vs mod 40）上的 AUC")
print("=" * 92)
print("  说明：你的方法是硬决策，转为打分 —— 有脉冲率记 1.0，无脉冲率记残余层后验 6/50=0.12")
print("        （即『二值后验打分』；这是对规则法最公平的连续化方式）\n")

prior = y[dd.hasPR.values == 0].mean()      # 残余层 mod 比例 = 6/50
S = {}
S["你的三层方法（二值后验：有脉冲=1，无脉冲=0.12）"] = np.where(dd.hasPR.values == 1, 1.0, prior)
S["你的方法 + 第三层 Fpeak 概率（连续化）"] = np.where(
    dd.hasPR.values == 1, 1.0, loo_prob(dd[["Fpeak"]].values.astype(float), y))
S["仅脉冲率有无（二值）"] = dd.hasPR.values.astype(float)
S["仅 Fpeak"] = dd.Fpeak.values.astype(float)
S["仅 F01"] = dd.F01.values.astype(float)
S["仅峰值位置（基频/谐波）"] = dd.peakpos.values.astype(float)
S["仅 Duration"] = dd.Duration.values.astype(float)
S["Fpeak + 脉冲率 逻辑回归"] = loo_prob(dd[["Fpeak", "hasPR"]].values.astype(float), y)
S["6 个声学量 逻辑回归"] = loo_prob(dd[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]].values.astype(float), y)
S["6 个声学量 + 脉冲率"] = loo_prob(
    dd[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "hasPR"]].values.astype(float), y)

rows = []
for k, v in S.items():
    a = auc(v, y)
    lo, hi = boot_ci(v, y)
    rows.append((k, a, lo, hi))
tab = pd.DataFrame(rows, columns=["打分方案", "AUC", "95%CI下限", "95%CI上限"]).sort_values(
    "AUC", ascending=False).reset_index(drop=True)
tab.insert(0, "排名", range(1, len(tab) + 1))
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 92)
print("二、为什么『加第三层』反而让 AUC 变低")
print("=" * 92)
res = dd.hasPR.values == 0
p_res = loo_prob(dd[["Fpeak"]].values.astype(float), y)[res]
yr = y[res]
print(f"  残余 50 条内：Fpeak 的 AUC = {auc(p_res, yr):.4f}")
print("  含义：在无脉冲率的样本里，Fpeak 的排序是**反向**的——")
print("        6 条 mod 的 P 值反而低于多数 flat。")
print("  → 规则法在这一层是『并列』（计 0.5），而加入 Fpeak 后变成『排错』（计 0）。")
print("    所以 AUC 上：规则法 0.925 > 加第三层 0.854。")

print("\n" + "=" * 92)
print("三、三分类（108 条）的一对多 AUC")
print("=" * 92)


def softmax_fit(X, yc, K, lr=.5, iters=5000, lam=1e-2):
    n, p = X.shape
    Y = np.eye(K)[yc]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = X @ W; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / n + lam * W)
    return W


def loo_prob_multi(X, yc, K):
    out = np.zeros((len(yc), K))
    for i in range(len(yc)):
        tr = np.ones(len(yc), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = softmax_fit((X[tr] - mu) / sd, yc[tr], K)
        Z = ((X[i] - mu) / sd) @ W
        e = np.exp(Z - Z.max()); out[i] = e / e.sum()
    return out


TYPES = ["flat", "mod", "tv"]
yc3 = np.array([TYPES.index(v) for v in d.CallType])
# 你的三层方法的三分类概率（结构性 + 唯一模糊处用先验）
P = np.zeros((len(d), 3))
for i, r in d.iterrows():
    if r.hasF02:
        P[i] = [0, 0, 1]
    elif r.hasPR:
        P[i] = [0, 1, 0]
    else:
        P[i] = [1 - prior, prior, 0]

for name, X in [("你的三层方法（概率化）", None),
                ("多项logit：Fpeak+脉冲率+F02", d[["Fpeak", "hasPR", "hasF02"]].values.astype(float)),
                ("多项logit：6 个声学量", d[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]].values.astype(float)),
                ("多项logit：6 声学量+脉冲率+F02", d[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "hasPR", "hasF02"]].values.astype(float))]:
    if X is not None:
        P = loo_prob_multi(X, yc3, 3)
    aucs = [auc(P[:, k], (yc3 == k).astype(int)) for k in range(3)]
    print(f"  {name:34s} flat={aucs[0]:.3f}  mod={aucs[1]:.3f}  tv={aucs[2]:.3f}  "
          f"macro={np.mean(aucs):.3f}")
