# -*- coding: utf-8 -*-
"""第一问混淆矩阵（已修正：逻辑回归含截距项）
   fig9 最终模型单图；fig10 四方案对比（含 8 特征上界模型）"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

D = r"X:\2026APMCM\_package\2026APMCM_C_P1\01_数据"
if not os.path.exists(os.path.join(D, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")):
    D = r"X:\2026APMCM\题目与要求\date"
OUT = r"X:\2026APMCM\_package\2026APMCM_C_P1\03_图表"
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(D, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
TYPES = ["flat", "mod", "tv"]
d["hasPR"] = d.PulseRate.notna().astype(int)
d["hasF02"] = d.F02.notna().astype(int)
y = np.array(d.CallType.tolist(), dtype=object)
yc = np.array([TYPES.index(v) for v in y])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def softmax_fit(X, yy, K, lr=.3, iters=8000, lam=1e-2):
    """含截距列的 softmax 回归"""
    Xb = np.column_stack([np.ones(len(X)), X])
    n, p = Xb.shape
    Y = np.eye(K)[yy]
    W = np.zeros((p, K))
    for _ in range(iters):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (Xb.T @ (P - Y) / n + lam * W)
    return W


def loo_multi(X, yy, K):
    pred = np.empty(len(yy), int)
    for i in range(len(yy)):
        tr = np.ones(len(yy), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        W = softmax_fit((X[tr] - mu) / sd, yy[tr], K)
        xi = np.r_[1, (X[i] - mu) / sd]
        pred[i] = int(np.argmax(xi @ W))
    return pred


feats6 = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
X6 = d[feats6].values.astype(float)
X3 = d[["Fpeak", "hasPR", "hasF02"]].values.astype(float)
X8 = d[feats6 + ["hasPR", "hasF02"]].values.astype(float)

preds = {}
preds["(a) 三层分层判别\nF02→tv，脉冲串→mod，其余→flat"] = np.where(
    d.hasF02 == 1, 2, np.where(d.hasPR == 1, 1, 0))
preds["(b) 多项 logit（3 特征）\nFpeak + 脉冲串 + F02"] = loo_multi(X3, yc, 3)
preds["(c) 多项 logit（仅声学）\n6 个连续声学量"] = loo_multi(X6, yc, 3)
preds["(d) 多项 logit（全特征，性能上界）\n6 声学量 + 脉冲串 + F02"] = loo_multi(X8, yc, 3)


def cm(pred):
    M = np.zeros((3, 3), int)
    for t, q in zip(yc, pred):
        M[t, q] += 1
    return M


def bal(pred):
    return np.mean([((pred == k) & (yc == k)).sum() / (yc == k).sum() for k in range(3)])


cmap = LinearSegmentedColormap.from_list("soft", ["#ffffff", "#d6e4f0", "#4C72B0", "#1f3b5c"])


def draw(ax, M, title, sub, fs_ann=15):
    row = M.sum(1, keepdims=True).astype(float)
    P = M / np.where(row == 0, 1, row)
    ax.imshow(P, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            if M[i, j] == 0:
                ax.text(j, i, "0", ha="center", va="center", color="#c8c8c8", fontsize=fs_ann - 3)
                continue
            col = "white" if P[i, j] > .55 else ("#C44E52" if i != j else "#1a1a1a")
            ax.text(j, i - .10, f"{M[i,j]}", ha="center", va="center", color=col,
                    fontsize=fs_ann, fontweight="bold")
            ax.text(j, i + .26, f"{P[i,j]*100:.0f}%", ha="center", va="center",
                    color=col, fontsize=fs_ann - 6)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(TYPES, fontsize=11); ax.set_yticklabels(TYPES, fontsize=11)
    ax.set_xlabel("预测类型", fontsize=10.5); ax.set_ylabel("真实类型", fontsize=10.5)
    ax.set_title(f"{title}\n{sub}", fontsize=10.5, pad=10)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="w", lw=2.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_edgecolor("#dddddd")


# ---------------- 图 9：最终模型
key_a = "(a) 三层分层判别\nF02→tv，脉冲串→mod，其余→flat"
M = cm(preds[key_a])
fig, ax = plt.subplots(figsize=(7.6, 5.1))
draw(ax, M, "三层分层判别 · 混淆矩阵",
     f"整体准确率 {M.trace()}/{M.sum()} = {M.trace()/M.sum()*100:.2f}%　平衡准确率 {bal(preds[key_a])*100:.1f}%",
     fs_ann=17)
ax.set_ylabel("真实类型", fontsize=10.5, labelpad=10)
ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec="#C44E52", lw=2.6))
ax.annotate("6 例 mod→flat\n脉冲串未检出\n（测量失败，非声学重叠）",
            xy=(0.42, 0.82), xytext=(3.05, 1.75), fontsize=8.8, color="#C44E52",
            linespacing=1.7, va="center",
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#C44E52",
                            connectionstyle="arc3,rad=0.28"))
ax.set_xlim(-.5, 4.35)
plt.tight_layout()
p9 = os.path.join(OUT, "fig9_混淆矩阵_最终模型.png")
plt.savefig(p9, dpi=210, bbox_inches="tight"); plt.close(fig)
print("[保存]", p9)

# ---------------- 图 10：四方案对比
fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.2))
for ax, (title, pred) in zip(axes.ravel(), preds.items()):
    M = cm(pred)
    draw(ax, M, title, f"准确率 {M.trace()/M.sum()*100:.2f}%（{M.trace()}/{M.sum()}）　"
                       f"平衡准确率 {bal(pred)*100:.1f}%", fs_ann=14)
plt.suptitle("第一问：四种方案的混淆矩阵对比\n"
             "底色为行归一化比例；格内上为样本数、下为行百分比；红色为辅对角错误",
             fontsize=12.5, y=1.005)
plt.tight_layout()
p10 = os.path.join(OUT, "fig10_混淆矩阵_四方案对比.png")
plt.savefig(p10, dpi=200, bbox_inches="tight"); plt.close(fig)
print("[保存]", p10)

print("\n各方案指标：")
for title, pred in preds.items():
    M = cm(pred)
    print(f"\n{title.replace(chr(10), ' | ')}")
    print(f"  准确率 {M.trace()/M.sum():.4f}   平衡准确率 {bal(pred):.4f}")
    print(pd.DataFrame(M, index=[f"真:{t}" for t in TYPES],
                       columns=[f"测:{t}" for t in TYPES]).to_string())
