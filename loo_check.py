# -*- coding: utf-8 -*-
"""共享连续特征的真实判别力：LOO 分类 + 置换零分布 + 单特征 AUC。纯 numpy。"""
import pandas as pd
import numpy as np

D = r"X:\2026APMCM\题目与要求\date"
d1 = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
feats = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
rng = np.random.default_rng(0)


def fit_softmax(X, y, K, lr=0.5, iters=1200, lam=1e-2):
    n, p = X.shape
    Y = np.eye(K)[y]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / n + lam * W)
    return W


def loo_acc(X, yc, K):
    correct = 0
    for i in range(len(yc)):
        tr = np.ones(len(yc), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = fit_softmax((X[tr] - mu) / sd, yc[tr], K)
        if np.argmax(((X[i] - mu) / sd) @ W) == yc[i]:
            correct += 1
    return correct / len(yc)


def auc(pos, neg):
    """Mann-Whitney AUC, 不需 scipy。"""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 平均秩处理并列
    s = pd.Series(allv).rank().values
    R1 = s[: len(pos)].sum()
    return (R1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


X = d1[feats].values
y = d1["CallType"].values
cls3 = ["flat", "mod", "tv"]
yc3 = np.array([cls3.index(v) for v in y])

print("=== 三分类，仅 6 个共享特征，LOO ===")
acc3 = loo_acc(X, yc3, 3)
print("LOO 准确率 = %.3f   多数类基线 = %.3f" % (acc3, np.bincount(yc3).max() / len(yc3)))

print("\n=== 置换零分布（30 次打乱标签，同样 LOO）===")
null = []
for k in range(30):
    yp = rng.permutation(yc3)
    null.append(loo_acc(X, yp, 3))
null = np.array(null)
print("零分布: 均值=%.3f  95%%分位=%.3f  最大=%.3f" % (null.mean(), np.quantile(null, .95), null.max()))
print("实际值在零分布中的位置: p = %.2f" % ((null >= acc3).mean()))

print("\n=== flat vs mod，仅共享特征，LOO ===")
m = np.isin(y, ["flat", "mod"])
Xm, ym = X[m], y[m]
ycm = np.array([0 if v == "flat" else 1 for v in ym])
acc2 = loo_acc(Xm, ycm, 2)
print("LOO 准确率 = %.3f   基线 = %.3f" % (acc2, max(ycm.mean(), 1 - ycm.mean())))

print("\n=== 单特征 AUC（0.5 = 无判别力）===")
pairs = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
for a, b in pairs:
    line = []
    for f in feats:
        va = d1.loc[d1.CallType == a, f].values
        vb = d1.loc[d1.CallType == b, f].values
        A = auc(va, vb)
        line.append(f"{f}={A:.2f}")
    print(f"  {a:4s} vs {b:4s}: " + "  ".join(line))

print("\n=== 结论性对照 ===")
print("  结构规则(F02有无 + PulseRate有无): 102/108 = 0.944")
print("  仅共享连续特征 三分类 LOO: %.3f (接近多数类基线)" % acc3)
