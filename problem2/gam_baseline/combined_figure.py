# -*- coding: utf-8 -*-
"""
一张图：声保真度退化曲线（作者 GAM × CRITIC 主权重）＋ 企鹅回放响应率（Data02）。

左轴 = 保真度 F（0–1）；右轴 = 回放响应率（0–1）。横轴 = 传播距离（对数）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA_DIR = Path(r"C:\Users\34232\Documents\GitHub\2026APMCM\题目与要求\date")
GAM_CSV = Path(r"G:\亚太数模\2026APMCM\problem2\results\gam_predictions.csv")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ["flat", "mod", "tv"]
METRICS = ["spectrogram", "spectrum", "envelope"]
FILE_GRID = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])


def setup_font():
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            fm.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(p)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_gam():
    df = pd.read_csv(GAM_CSV)
    full = df[df.hold == 0]
    grid = np.sort(full.d.unique())
    gam = np.empty((3, 3, len(grid)))
    for ci, c in enumerate(CALLS):
        for mi, m in enumerate(METRICS):
            gam[ci, mi] = full[(full.call == c) & (full.metric == m)].sort_values("d").estimate.to_numpy()
    return grid, gam


def critic_weights(X):
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    sigma = Xn.std(axis=0, ddof=0)
    r = np.corrcoef(Xn, rowvar=False)
    conflict = (1 - r).sum(axis=1)
    return (sigma * conflict) / (sigma * conflict).sum()


def main():
    setup_font()
    grid, gam = load_gam()

    # CRITIC 主权重
    means = np.zeros((3, 3, 7))
    for mi in range(3):
        for ci in range(3):
            (p,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(p)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            means[ci, mi] = df.astype(float).mean(axis=0).to_numpy()
    w = critic_weights(means.transpose(0, 2, 1).reshape(-1, 3))
    F = np.einsum("m,cmd->cd", w, gam)          # (3, N) CRITIC 加权保真度

    # Data02 回放响应率（按距离）
    pb = pd.read_csv(DATA_DIR / "Data02_AfricanPenguins_playbacks_data.csv")
    calls = pb[pb.Type == "call"]
    ctrl = pb[pb.Type == "control"]
    rc = calls.groupby("Distance").ResponseYesNo.agg(mean="mean", n="count")
    rc0 = ctrl.groupby("Distance").ResponseYesNo.agg(mean="mean", n="count")

    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)
    # 左轴：保真度
    for ci, c in enumerate(CALLS):
        ax.plot(grid, F[ci], color=colors[c], lw=2.4, label=f"{labels[c]} 保真度")
    for t in (0.5, 0.7):
        ax.axhline(t, color="gray", ls=":", lw=1)
    ax.text(4.1, 0.505, "潜在识别 τ=0.5", fontsize=9, va="bottom", color="gray")
    ax.text(4.1, 0.705, "可靠识别 τ=0.7", fontsize=9, va="bottom", color="gray")
    ax.set_xscale("log")
    ax.set_xticks(DIST); ax.set_xticklabels([str(int(d)) for d in DIST])
    ax.set_xlabel("传播距离 distance (m)")
    ax.set_ylabel("声保真度 F（CRITIC 主权重）", color="k")
    ax.set_ylim(0, 1)
    ax.grid(ls=":", alpha=0.4)

    # 右轴：回放响应率
    ax2 = ax.twinx()
    ax2.scatter(rc.index, rc["mean"], s=60, color="#C00000", marker="o",
                zorder=5, edgecolor="white", label="自然声响应率")
    ax2.scatter(rc0.index, rc0["mean"], s=42, color="gray", marker="x",
                zorder=5, label="对照声响应率")
    # 样本数标注
    for d, row in rc.iterrows():
        ax2.annotate(f"n={int(row.n)}", (d, row["mean"]), textcoords="offset points",
                     xytext=(0, 9), fontsize=7, ha="center", color="#C00000")
    ax2.set_ylabel("回放响应率（Data02）", color="#C00000")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="#C00000")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="center right")
    ax.set_title("声保真度衰减 vs 企鹅回放响应（作者 GAM × CRITIC + Data02）")
    fig.tight_layout()
    fig.savefig(OUT / "fidelity_vs_playback.png")
    print(f"已保存：{OUT / 'fidelity_vs_playback.png'}")

    # 顺便打印响应率数值
    print("\n各距离自然声/对照响应率：")
    for d in sorted(rc.index):
        ctrl_rate = rc0.loc[d, "mean"] if d in rc0.index else float("nan")
        print(f"  {int(d):>3} m  自然声 {rc.loc[d,'mean']:.0%} (n={int(rc.loc[d,'n'])})   对照 {ctrl_rate:.0%}")


if __name__ == "__main__":
    main()
