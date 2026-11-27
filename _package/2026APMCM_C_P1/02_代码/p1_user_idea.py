# -*- coding: utf-8 -*-
"""
检验用户思路：用 Fpeak + 脉冲率 做逻辑回归分类器。
对比 4 种模型：
  A  两特征（Fpeak, 脉冲率0编码）        三分类
  B  两特征                              flat vs mod 二分类
  C  加 F02 有无（三特征）                三分类
  D  全量结构+声学（上界）
并给出两阶段模型的表现。
"""
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
TYPES = ["flat", "mod", "tv"]

d["hasF02"] = d.F02.notna().astype(float)
d["hasPR"] = d.PulseRate.notna().astype(float)
d["PR0"] = d.PulseRate.fillna(0.0)                        # 无脉冲串编码为 0
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(float)         # 1 = 峰值落在基频
d["BW"] = d.Q75 - d.Q25
y3 = np.array([TYPES.index(v) for v in d.CallType])


def softmax_fit(X, yc, K, lr=.5, iters=4000, lam=1e-2):
    n, p = X.shape
    Y = np.eye(K)[yc]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / n + lam * W)
    return W


def loo(X, yc, K):
    pred = np.empty(len(yc), int)
    for i in range(len(yc)):
        tr = np.ones(len(yc), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = softmax_fit((X[tr] - mu) / sd, yc[tr], K)
        pred[i] = np.argmax(((X[i] - mu) / sd) @ W)
    return pred


def report(tag, X, yc, labels):
    K = len(labels)
    pred = loo(X, yc, K)
    acc = (pred == yc).mean()
    base = np.bincount(yc).max() / len(yc)
    print(f"\n【{tag}】LOO 准确率 = {acc:.3f}   多数类基线 = {base:.3f}")
    ct = pd.crosstab(pd.Series([labels[i] for i in yc], name="真实"),
                     pd.Series([labels[i] for i in pred], name="预测"))
    print(ct.to_string())
    return acc, pred


print("=" * 78)
print("一、你的方案：Fpeak + 脉冲率（三分类）")
print("=" * 78)
X_A = d[["Fpeak", "PR0"]].values
report("A  Fpeak + 脉冲率0编码", X_A, y3, TYPES)

print("\n" + "=" * 78)
print("二、只分 flat vs mod：这两个够不够？")
print("=" * 78)
m2 = (d.CallType != "tv").values
y2 = (d.CallType[m2] == "mod").astype(int).values
accB, predB = report("B  Fpeak + 脉冲率（flat vs mod）", X_A[m2], y2, ["flat", "mod"])

print("\n" + "=" * 78)
print("三、加上 F02 有无（三分类）")
print("=" * 78)
X_C = d[["Fpeak", "PR0", "hasF02"]].values
report("C  Fpeak + 脉冲率 + F02有无", X_C, y3, TYPES)

print("\n" + "=" * 78)
print("四、上界参考：6 个声学量 + 两个结构量")
print("=" * 78)
X_D = d[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "hasPR", "hasF02"]].values
report("D  全量", X_D, y3, TYPES)

print("\n" + "=" * 78)
print("五、两阶段模型（推荐架构）")
print("=" * 78)
# 阶段1: flat vs (mod/tv)
y_s1 = (d.CallType != "flat").astype(int).values
acc_s1, _ = report("阶段1  flat vs 非flat（Fpeak + 脉冲率）", X_A, y_s1, ["flat", "notflat"])
# 阶段2: mod vs tv（仅结构量）
sub = (d.CallType != "flat").values
X_s2 = d.loc[sub, ["hasF02"]].values
y_s2 = (d.CallType[sub] == "tv").astype(int).values
pred_s2 = loo(X_s2, y_s2, 2)
acc_s2 = (pred_s2 == y_s2).mean()
print(f"\n【阶段2  mod vs tv（F02 有无）】LOO 准确率 = {acc_s2:.3f}")
overall = acc_s1 * 0.667 + acc_s2 * 0.333     # 按类占比加权近似
print(f"  两阶段总体准确率 ≈ {overall:.3f}（按 flat 41% / mod 37% / tv 22% 加权）")

print("\n" + "=" * 78)
print("六、系数解释（全样本拟合，标准化后，仅作方向与量级参考）")
print("=" * 78)
Xs = X_A.copy()
Xs = (Xs - Xs.mean(0)) / Xs.std(0)
W = softmax_fit(Xs, y3, 3, iters=8000, lam=1e-3)
print(f"{'特征':10s}{'→flat':>10s}{'→mod':>10s}{'→tv':>10s}")
for i, f in enumerate(["Fpeak", "脉冲率"]):
    print(f"{f:10s}" + "".join(f"{W[i, k]:10.3f}" for k in range(3)))

print("\n" + "=" * 78)
print("七、单特征在 flat vs mod 上的 AUC")
print("=" * 78)
def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    R1 = r[:len(pos)].sum()
    return (R1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for f in ["Fpeak", "F01", "PR0", "peakpos"]:
    va = d.loc[(d.CallType == "flat"), f].values
    vb = d.loc[(d.CallType == "mod"), f].values
    print(f"  {f:10s} AUC = {auc(vb, va):.3f}   (0.5=随机, 1=完美)")
