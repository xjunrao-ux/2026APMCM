# -*- coding: utf-8 -*-
"""三层分类器：L1 F02→tv；L2 有脉冲率→mod；L3 剩余 50 条用 Fpeak 等做逻辑回归。
并对比：三行硬规则 / 三特征多项logit。全部 LOO 评价。"""
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"
d = pd.read_csv(D + r"\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
TYPES = ["flat", "mod", "tv"]
d["hasF02"] = d.F02.notna().astype(int)
d["hasPR"] = d.PulseRate.notna().astype(int)
d["PR0"] = d.PulseRate.fillna(0.0)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
d["BW"] = d.Q75 - d.Q25
y = np.array(d.CallType.tolist(), dtype=object)


def softmax_fit(X, yc, K, lr=.5, iters=6000, lam=1e-2):
    n, p = X.shape
    Y = np.eye(K)[yc]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / n + lam * W)
    return W


def loo_pred(X, yc, K):
    pred = np.empty(len(yc), int)
    for i in range(len(yc)):
        tr = np.ones(len(yc), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = softmax_fit((X[tr] - mu) / sd, yc[tr], K)
        pred[i] = np.argmax(((X[i] - mu) / sd) @ W)
    return pred


def show(tag, ytrue, ypred, labels):
    acc = (ypred == np.array([labels.index(v) for v in ytrue])).mean()
    print(f"  {tag}: 准确率 = {acc:.4f} ({(np.array([labels.index(v) for v in ytrue])==ypred).sum()}/{len(ytrue)})")
    return acc


print("=" * 80)
print("一、逐层看：哪些样本落到下一层")
print("=" * 80)
l1_tv = d.hasF02 == 1
l2_mod = (~l1_tv) & (d.hasPR == 1)
l3 = (~l1_tv) & (~l2_mod)
print(f"  L1 有F02 → 判 tv        : {l1_tv.sum():3d} 条，真实分布 {d[l1_tv].CallType.value_counts().to_dict()}")
print(f"  L2 有脉冲率 → 判 mod    : {l2_mod.sum():3d} 条，真实分布 {d[l2_mod].CallType.value_counts().to_dict()}")
print(f"  L3 剩余                 : {l3.sum():3d} 条，真实分布 {d[l3].CallType.value_counts().to_dict()}")
print(f"\n  ★ L3 这 {l3.sum()} 条的脉冲率可用数 = {d.loc[l3,'PulseRate'].notna().sum()}"
      f"  → 脉冲率在这一层是常数，无法作为特征")

print("\n" + "=" * 80)
print("二、三层分类器整体表现（L3 用 LOO 逻辑回归）")
print("=" * 80)
res = {}
for feats3, name in [(["Fpeak"], "Fpeak"),
                     (["Fpeak", "peakpos"], "Fpeak+峰值位置"),
                     (["Fpeak", "F01", "peakpos", "BW"], "Fpeak+F01+峰值位置+带宽"),
                     (["Fpeak", "PR0"], "Fpeak+脉冲率(用户原设想)")]:
    pred = np.empty(len(d), dtype=object)
    pred[l1_tv.values] = "tv"
    pred[l2_mod.values] = "mod"
    sub = l3.values
    X3 = d.loc[sub, feats3].values.astype(float)
    yc3 = np.array([0 if v == "flat" else 1 for v in d.loc[sub, "CallType"]])
    p3 = loo_pred(X3, yc3, 2)
    pred[sub] = np.where(p3 == 0, "flat", "mod")
    correct = (pred == y).sum()
    acc = correct / len(y)
    print(f"\n  L3 特征 = {name:26s} 整体 = {correct}/{len(y)} = {acc:.4f}")
    ct = pd.crosstab(pd.Series(y, name="真实"), pd.Series(pred, name="预测"))
    print(ct.to_string().replace("\n", "\n  "))
    # L3 内部指标
    y3t = d.loc[sub, "CallType"].values
    y3p = pred[sub]
    tp = ((y3p == "mod") & (y3t == "mod")).sum()
    fp = ((y3p == "mod") & (y3t == "flat")).sum()
    fn = ((y3p == "flat") & (y3t == "mod")).sum()
    print(f"  L3 内部: 抓回 mod {tp}/6，误判 flat {fp}，漏掉 {fn}；"
          f"平衡准确率 = {0.5*(tp/6 + 1-fp/44):.3f}（vs 全判flat基线 {44/50:.3f}）")
    res[name] = acc

print("\n" + "=" * 80)
print("三、对照：其他两种做法")
print("=" * 80)
# 三行硬规则
pred_rule = np.where(d.hasF02 == 1, "tv", np.where(d.hasPR == 1, "mod", "flat"))
print(f"  三行硬规则（L3 全判 flat）: {(pred_rule==y).sum()}/{len(y)} = {(pred_rule==y).mean():.4f}")

# 三特征多项 logit
X3f = d[["Fpeak", "hasPR", "hasF02"]].values.astype(float)
yc3 = np.array([TYPES.index(v) for v in y])
p3f = loo_pred(X3f, yc3, 3)
print(f"  三特征多项 logit（Fpeak+脉冲串+F02）: {(p3f==yc3).sum()}/{len(y)} = {(p3f==yc3).mean():.4f}")
print(pd.crosstab(pd.Series([TYPES[i] for i in yc3], name="真实"),
                  pd.Series([TYPES[i] for i in p3f], name="预测")).to_string())

print("\n" + "=" * 80)
print("四、最理想的第三层：如果脉冲串能测出来")
print("=" * 80)
print("  6 条缺脉冲率的 mod，若脉冲串可测，则它们会被 L2 正确抓走：")
print(f"  理论上限 = {(l1_tv.sum() + 40 + 44)}/{len(y)} = {(l1_tv.sum()+40+44)/len(y):.4f}")
print("  → 当前 0.94~0.95 与上限 1.00 的差距，全部来自这 6 次测量失败")
