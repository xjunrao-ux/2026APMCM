# -*- coding: utf-8 -*-
"""按用户最新方案：对 84 条(flat 44 + mod 40)做回归，再用"检出脉冲率→mod"覆盖，看准确率。
对比多种训练集/特征/是否覆盖的组合。"""
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
d["hasF02"] = d.F02.notna().astype(int)
d["hasPR"] = d.PulseRate.notna().astype(int)
d["PR0"] = d.PulseRate.fillna(0.0)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
d["BW"] = d.Q75 - d.Q25


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit_fit(X, y, lr=.5, iters=5000, lam=1e-2):
    n, p = X.shape
    w = np.zeros(p + 1)
    Xb = np.column_stack([np.ones(n), X])
    for _ in range(iters):
        p_ = sigmoid(Xb @ w)
        g = Xb.T @ (p_ - y) / n + lam * np.r_[0, w[1:]]
        w -= lr * g
    return w


def logit_pred(w, X):
    return sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)


def loo84(feats, override):
    """在 84 条上 LOO；返回 84 条的预测(0=flat,1=mod)"""
    sub = (d.CallType != "tv").values
    dd = d[sub].reset_index(drop=True)
    X = dd[feats].values.astype(float)
    y = (dd.CallType == "mod").astype(int).values
    pred = np.empty(len(y), int)
    prob = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, y[tr])
        pr = logit_pred(w, ((X[i] - mu) / sd)[None, :])[0]
        prob[i] = pr
        pred[i] = int(pr > .5)
        if override and dd.hasPR.iloc[i] == 1:
            pred[i] = 1
    return y, pred, prob, dd


def overall(y84, pred84):
    """tv 层 24 条全对，合计 108"""
    correct84 = (y84 == pred84).sum()
    return correct84, correct84 + 24, (correct84 + 24) / 108


print("=" * 84)
print("一、基线：硬规则（84 条内：有脉冲率→mod，否则 flat）")
print("=" * 84)
sub = (d.CallType != "tv").values
dd = d[sub]
y84_true = (dd.CallType == "mod").astype(int).values
pred_rule = dd.hasPR.values
print(f"  84 条内正确 {int((pred_rule==y84_true).sum())}/84 = {(pred_rule==y84_true).mean():.4f}")
print(f"  整体 {int((pred_rule==y84_true).sum())+24}/108 = {((pred_rule==y84_true).sum()+24)/108:.4f}")

print("\n" + "=" * 84)
print("二、你的方案：84 条训练回归 + 脉冲率覆盖")
print("=" * 84)
rows = []
for feats, name in [(["Fpeak"], "Fpeak"),
                    (["Fpeak", "hasPR"], "Fpeak+脉冲率"),
                    (["Fpeak", "PR0"], "Fpeak+脉冲率(数值)"),
                    (["Fpeak", "hasPR", "peakpos"], "Fpeak+脉冲率+峰值位置"),
                    (["Fpeak", "hasPR", "peakpos", "BW", "F01"], "全部声学+脉冲率")]:
    for ov in [False, True]:
        y, pred, prob, _ = loo84(feats, ov)
        c84, c_all, acc = overall(y, pred)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        rows.append(dict(特征=name, 覆盖="是" if ov else "否",
                         n84正确=c84, 整体正确=c_all, 整体准确率=acc,
                         mod召回=f"{tp}/40", flat误判=fp))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 84)
print("三、关键：覆盖之后，剩下 44 flat + 6 mod 由回归怎么判")
print("=" * 84)
y, pred, prob, dd = loo84(["Fpeak", "hasPR"], True)
rest = dd.hasPR.values == 0
print(f"  残余样本 {rest.sum()} 条（flat 44 + mod 6）")
print(f"  其中被判 mod 的: {int((pred[rest]==1).sum())} 条；"
      f"真实 mod 被抓回 {int(((pred[rest]==1)&(y[rest]==1)).sum())}/6")
print(f"  回归对残余样本的判别力 AUC: ", end="")
def auc(pos, neg):
    a = np.concatenate([pos, neg]); r = pd.Series(a).rank().values
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
print(f"{auc(prob[rest][y[rest]==1], prob[rest][y[rest]==0]):.3f}  (0.5=无判别力)")

print("\n  那 6 条 mod 的真实 Fpeak 与回归概率：")
for i in np.where(rest & (y == 1))[0]:
    print(f"    Fpeak={dd.Fpeak.iloc[i]:7.0f}  F01={dd.F01.iloc[i]:5.0f}  P(mod)={prob[i]:.3f}")
print("\n  flat 中被误判为 mod 的：")
for i in np.where(rest & (y == 0) & (pred == 1))[0]:
    print(f"    Fpeak={dd.Fpeak.iloc[i]:7.0f}  P(mod)={prob[i]:.3f}")

print("\n" + "=" * 84)
print("四、残余层用 Fpeak 有没有可能做好？穷举所有阈值")
print("=" * 84)
f = dd.Fpeak.values[rest]; t = y[rest]
order = np.argsort(f)
best = None
for thr in np.concatenate([[-1e9], f[order], [1e9]]):
    for side in ["lt", "gt"]:
        pr = (f < thr) if side == "lt" else (f > thr)
        c = (pr.astype(int) == t).sum()
        if best is None or c > best[0]:
            best = (c, thr, side, pr.sum())
print(f"  残余 50 条的最佳单阈值规则: 正确 {best[0]}/50，阈值 {best[1]:.0f} Hz，方向 {best[2]}，判 mod {best[3]} 条")
print(f"  → 整体最好也只是 {(best[0]+58)}/108 = {(best[0]+58)/108:.4f}")
