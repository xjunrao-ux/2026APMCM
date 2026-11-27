# -*- coding: utf-8 -*-
"""fig9 单独重绘（不重算模型，速度快）：修复 y 轴标签被裁切"""
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
OUT = r"X:\2026APMCM\_package\2026APMCM_C_P1\03_图表"
d = pd.read_csv(os.path.join(D, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
TYPES = ["flat", "mod", "tv"]
pred = np.where(d.F02.notna(), 2, np.where(d.PulseRate.notna(), 1, 0))
yc = np.array([TYPES.index(v) for v in d.CallType])
M = np.zeros((3, 3), int)
for t, q in zip(yc, pred):
    M[t, q] += 1
bal = np.mean([((pred == k) & (yc == k)).sum() / (yc == k).sum() for k in range(3)])

cmap = LinearSegmentedColormap.from_list("soft", ["#ffffff", "#d6e4f0", "#4C72B0", "#1f3b5c"])
fig, ax = plt.subplots(figsize=(8.0, 5.0))
row = M.sum(1, keepdims=True).astype(float)
P = M / row
ax.imshow(P, cmap=cmap, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        if M[i, j] == 0:
            ax.text(j, i, "0", ha="center", va="center", color="#c8c8c8", fontsize=14)
            continue
        col = "white" if P[i, j] > .55 else ("#C44E52" if i != j else "#1a1a1a")
        ax.text(j, i - .10, f"{M[i,j]}", ha="center", va="center", color=col,
                fontsize=17, fontweight="bold")
        ax.text(j, i + .26, f"{P[i,j]*100:.0f}%", ha="center", va="center",
                color=col, fontsize=11)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(TYPES, fontsize=11); ax.set_yticklabels(TYPES, fontsize=11)
ax.set_xlabel("预测类型", fontsize=10.5)
ax.set_ylabel("真实类型", fontsize=10.5, labelpad=14)
ax.set_title("三层分层判别 · 混淆矩阵\n"
             f"整体准确率 {M.trace()}/{M.sum()} = {M.trace()/M.sum()*100:.2f}%　"
             f"平衡准确率 {bal*100:.1f}%", fontsize=11, pad=12)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="w", lw=2.2)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_edgecolor("#dddddd")
ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec="#C44E52", lw=2.6))
ax.annotate("6 例 mod→flat\n脉冲串未检出\n（测量失败，非声学重叠）",
            xy=(0.42, 0.85), xytext=(3.15, 1.75), fontsize=8.8, color="#C44E52",
            linespacing=1.7, va="center",
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#C44E52",
                            connectionstyle="arc3,rad=0.28"))
ax.set_xlim(-.5, 4.45)
plt.subplots_adjust(left=.13, right=.99, bottom=.12, top=.88)
p = os.path.join(OUT, "fig9_混淆矩阵_最终模型.png")
plt.savefig(p, dpi=210); plt.close(fig)
print("[保存]", p, "| 混淆矩阵:", M.tolist())
