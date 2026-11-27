# -*- coding: utf-8 -*-
"""打包步骤 1：从 6ec2d13 提取全部数据到包目录；步骤 2：生成描述统计表与总览图。"""
import os
import subprocess
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

REPO = r"X:\2026APMCM"
PKG = os.path.join(REPO, "_package", "2026APMCM_C_P1")
D_DAT = os.path.join(PKG, "01_数据")
D_FIG = os.path.join(PKG, "03_图表")
for p in [PKG, D_DAT, D_FIG]:
    os.makedirs(p, exist_ok=True)

# ---------------- 提取数据
COMMIT, SUB = "6ec2d13", "题目与要求/date"
r = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "--name-only",
                    COMMIT, SUB + "/"], cwd=REPO, capture_output=True)
names = r.stdout.decode("utf-8", "replace").splitlines()
n_ok = 0
for n in names:
    base = os.path.basename(n)
    if base.lower().endswith(".mp4"):
        continue
    dst = os.path.join(D_DAT, base)
    d = subprocess.run(["git", "-c", "core.quotepath=false", "show", f"{COMMIT}:{n}"],
                       cwd=REPO, capture_output=True)
    if d.returncode == 0:
        with open(dst, "wb") as f:
            f.write(d.stdout)
        n_ok += 1
print(f"提取数据 {n_ok} 个文件 -> {D_DAT}")

# ---------------- 描述统计
d = pd.read_csv(os.path.join(D_DAT, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
TYPES = ["flat", "mod", "tv"]
d["BW"] = d.Q75 - d.Q25
d["PulseCount"] = d.Duration * d.PulseRate
d["SpectralSkew"] = (d.Q75 + d.Q25 - 2 * d.Q50) / (d.Q75 - d.Q25)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)

VAR = [("Duration", "时长 (s)"), ("F01", "第一基频 F01 (Hz)"), ("F02", "第二基频 F02 (Hz)"),
       ("Fpeak", "峰值频率 Fpeak (Hz)"), ("Q25", "频谱 Q25 (Hz)"), ("Q50", "频谱 Q50 (Hz)"),
       ("Q75", "频谱 Q75 (Hz)"), ("PulseRate", "脉冲率 (次/s)"),
       ("BW", "带宽 Q75−Q25 (Hz)"), ("PulseCount", "脉冲数"), ("SpectralSkew", "谱偏度")]

rows = []
for col, lab in VAR:
    for t in TYPES:
        s = d.loc[d.CallType == t, col].dropna()
        if len(s) == 0:
            rows.append([lab, t, 0, "", "", "", "", "", "", ""])
            continue
        rows.append([lab, t, len(s), round(s.mean(), 3), round(s.std(), 3),
                     round(s.median(), 3),
                     f"[{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]",
                     round(s.std() / s.mean(), 3) if s.mean() else np.nan,
                     round(s.min(), 3), round(s.max(), 3)])
tab = pd.DataFrame(rows, columns=["指标", "类型", "n", "均值", "标准差", "中位数", "四分位区间", "CV", "最小", "最大"])
csv_out = os.path.join(PKG, "04_报告", "_表1_描述统计.csv")
os.makedirs(os.path.dirname(csv_out), exist_ok=True)
tab.to_csv(csv_out, index=False, encoding="utf-8-sig")
print(f"描述统计表已存 {csv_out}")
print(tab.to_string(index=False))

# ---------------- 图：多面板
fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.4))
COL = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}
panels = [("Duration", "时长 (s)"), ("F01", "F01 (Hz)"), ("Fpeak", "Fpeak (Hz)"),
          ("Q25", "Q25 (Hz)"), ("Q50", "Q50 (Hz)"), ("Q75", "Q75 (Hz)"),
          ("BW", "带宽 Q75−Q25 (Hz)"), ("PulseRate", "脉冲率 (次/s)")]
for ax, (col, lab) in zip(axes.ravel(), panels):
    data = [d.loc[d.CallType == t, col].dropna().values for t in TYPES]
    pos = [i for i, v in enumerate(data) if len(v) > 0]
    vp = ax.violinplot([data[i] for i in pos], positions=pos, widths=.75,
                       showmedians=True, showextrema=False)
    for b, i in zip(vp["bodies"], pos):
        b.set_facecolor(COL[TYPES[i]]); b.set_alpha(.55); b.set_edgecolor("w")
    vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.4)
    if len(pos) < 3:
        for i in range(3):
            if i not in pos:
                ax.text(i, ax.get_ylim()[1] * .5, "n=0\n(无脉冲串)", ha="center",
                        fontsize=8, color=COL[TYPES[i]],
                        bbox=dict(boxstyle="round,pad=.25", fc="white", ec=COL[TYPES[i]], ls="--"))
    ax.set_xticks(range(3)); ax.set_xticklabels(TYPES, fontsize=9)
    ax.set_ylabel(lab, fontsize=9); ax.grid(axis="y", alpha=.22)
plt.suptitle("图 1  三类海面鸣声的声学指标分布（小提琴 = 核密度，横线 = 中位数）", fontsize=12)
plt.tight_layout()
p1 = os.path.join(D_FIG, "fig1_描述统计_多指标分布.png")
plt.savefig(p1, dpi=180, bbox_inches="tight"); plt.close(fig)
print("[保存]", p1)
