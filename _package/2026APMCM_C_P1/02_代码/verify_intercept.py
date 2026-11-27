# -*- coding: utf-8 -*-
"""核对：softmax/logistic 带截距 vs 不带截距，对第一问各模型准确率的影响。"""
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\_package\2026APMCM_C_P1\01_数据"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
TYPES = ["flat", "mod", "tv"]
d["hasPR"] = d.PulseRate.notna().astype(int)
d["hasF02"] = d.F02.notna().astype(int)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
d["BW"] = d.Q75 - d.Q25
yc = np.array([TYPES.index(v) for v in d.CallType])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit(X, yy, K, intercept=True, lr=.3, iters=8000, lam=1e-2):
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    n, p = X.shape
    Y = np.eye(K)[yy]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = X @ W; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / n + lam * W)
    return W


def pred1(W, x, intercept=True):
    if intercept:
        x = np.r_[1, x]
    return int(np.argmax(x @ W))


def loo(X, yy, K, intercept=True):
    out = np.empty(len(yy), int)
    for i in range(len(yy)):
        tr = np.ones(len(yy), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = fit((X[tr] - mu) / sd, yy[tr], K, intercept)
        out[i] = pred1(W, (X[i] - mu) / sd, intercept)
    return out


def acc(p): return (p == yc).mean()


def bal(p):
    return np.mean([((p == k) & (yc == k)).sum() / (yc == k).sum() for k in range(3)])


feats6 = d[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]].values.astype(float)
X_b = d[["Fpeak", "hasPR", "hasF02"]].values.astype(float)
X_c = feats6

print("=" * 72)
print("三分类模型：带截距 vs 不带截距（LOO）")
print("=" * 72)
print(f"{'模型':34s}{'不带截距':>10s}{'带截距':>10s}{'差异':>9s}")
for name, X in [("6 个连续声学量", X_c), ("Fpeak+脉冲串+F02", X_b),
                ("6 声学量+脉冲串+F02", np.column_stack([feats6, d.hasPR, d.hasF02]))]:
    a0 = acc(loo(X, yc, 3, intercept=False))
    a1 = acc(loo(X, yc, 3, intercept=True))
    print(f"{name:34s}{a0:10.4f}{a1:10.4f}{a1-a0:+9.4f}")

print("\n带截距版本的详细结果：")
for name, X in [("6 个连续声学量", X_c), ("Fpeak+脉冲串+F02", X_b)]:
    p = loo(X, yc, 3, True)
    print(f"\n  {name}: 准确率 {acc(p):.4f}, 平衡准确率 {bal(p):.4f}")
    print(pd.crosstab(pd.Series([TYPES[i] for i in yc], name="真实"),
                      pd.Series([TYPES[i] for i in p], name="预测")).to_string())

print("\n" + "=" * 72)
print("第三层（残余 50 条）二分类：带截距 vs 不带截距")
print("=" * 72)
sub = (d.CallType != "tv").values
dd = d[sub].reset_index(drop=True)
y2 = (dd.CallType == "mod").astype(int).values


def loo2(X, yy, intercept=True):
    out = np.zeros(len(yy), int)
    for i in range(len(yy)):
        tr = np.ones(len(yy), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = fit((X[tr] - mu) / sd, yy[tr], 2, intercept)
        out[i] = pred1(W, (X[i] - mu) / sd, intercept)
    return out


# 84 条上（flat+mod）
for name, X in [("仅 Fpeak", dd[["Fpeak"]].values.astype(float)),
                ("Fpeak+脉冲串", dd[["Fpeak", "hasPR"]].values.astype(float))]:
    a0 = (loo2(X, y2, False) == y2).mean()
    a1 = (loo2(X, y2, True) == y2).mean()
    print(f"  84条 {name:16s} 无截距 {a0:.4f}  有截距 {a1:.4f}")

# 残余 50 条
res = np.where(dd.hasPR.values == 0)[0]
Xr = dd.loc[res, ["Fpeak"]].values.astype(float)
yr = y2[res]
for name, X, yy in [("残余50 Fpeak", Xr, yr),
                    ("残余50 Fpeak+峰值位置", dd.loc[res, ["Fpeak", "peakpos"]].values.astype(float), yr)]:
    p0 = loo2(X, yy, False); p1 = loo2(X, yy, True)
    print(f"  {name:22s} 无截距 抓回{int(((p0==1)&(yy==1)).sum())}/6 误判flat{int(((p0==1)&(yy==0)).sum())}"
          f" | 有截距 抓回{int(((p1==1)&(yy==1)).sum())}/6 误判flat{int(((p1==1)&(yy==0)).sum())}")
