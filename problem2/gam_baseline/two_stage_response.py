# -*- coding: utf-8 -*-
"""
两阶段回应模型：在分级阈值下游接一个"带地板"的回应概率，解释"200m 仍回应"。

    P(回应) = P(识别|F) · p_rec + (1 − P(识别|F)) · p_base
    P(识别|F) = sigmoid 软阈值 = 1 / (1 + e^{−k(F−τ)})

    τ     = 0.5   （潜在识别中点，阈值方案不动）
    k     = 20    （sigmoid 陡度，越大越接近硬阈值）
    p_rec = 0.647 （识别后的回应率 ≈ 自然声实测回应率）
    p_base= 0.129 （未识别时的基线回应率 ≈ 对照声实测回应率）

关键：F 低于 τ 时 P(识别)→0，但第二项 (1−P(识别))·p_base ≈ 0.129 给出"地板回应"，
     数值上自然解释"200m（F≈0.4 略低于 0.5）仍有回应"，无需改阈值。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA_DIR = Path(r"G:\亚太数模\2026APMCM\题目与要求\date")
GAM_CSV = Path(r"G:\亚太数模\2026APMCM\problem2\results\gam_predictions.csv")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ["flat", "mod", "tv"]
METRICS = ["spectrogram", "spectrum", "envelope"]
FILE_GRID = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])

TAU, K, P_REC, P_BASE = 0.5, 20.0, 0.647, 0.129


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
    return (sigma * (1 - r).sum(axis=1)) / (sigma * (1 - r).sum(axis=1)).sum()


def sigmoid(F, tau=TAU, k=K):
    return 1.0 / (1.0 + np.exp(-k * (F - tau)))


def response_rate(F, tau=TAU, k=K, p_rec=P_REC, p_base=P_BASE):
    P = sigmoid(F, tau, k)
    return P * p_rec + (1 - P) * p_base


def main():
    setup_font()
    grid, gam = load_gam()
    means = np.zeros((3, 3, 7))
    for mi in range(3):
        for ci in range(3):
            (p,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(p)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            means[ci, mi] = df.astype(float).mean(axis=0).to_numpy()
    w = critic_weights(means.transpose(0, 2, 1).reshape(-1, 3))
    F = np.einsum("m,cmd->cd", w, gam)          # (3, N) CRITIC 加权保真度

    # 在实测回放距离上算 F 和 P(回应)
    print(f"两阶段模型参数：τ={TAU}, k={K}, p_rec={P_REC}, p_base={P_BASE}")
    print(f"\nCRITIC 权重：声谱图 {w[0]:.3f} / 幅度谱 {w[1]:.3f} / 包络 {w[2]:.3f}\n")
    print("实测回放距离上的 F 与预测回应率 P(回应)：")
    print(f"{'距离':<8}{'叫声':<6}{'F':>8}{'P(识别|F)':>10}{'P(回应)':>10}")
    play_dists = [50, 100, 150, 200]
    rows = []
    for d in play_dists:
        for ci, c in enumerate(CALLS):
            f = float(np.interp(d, grid, F[ci]))
            pr = response_rate(f)
            rows.append(dict(distance=d, call=c, F=f, P_ident=sigmoid(f), P_response=pr))
            print(f"{d:<8}{c:<6}{f:>8.3f}{sigmoid(f):>10.3f}{pr:>10.3f}")
    pd.DataFrame(rows).to_csv(OUT / "two_stage_response.csv", index=False, encoding="utf-8-sig")

    # 图
    pb = pd.read_csv(DATA_DIR / "Data02_AfricanPenguins_playbacks_data.csv")
    calls = pb[pb.Type == "call"]
    ctrl = pb[pb.Type == "control"]
    rc = calls.groupby("Distance").ResponseYesNo.mean()
    rc0 = ctrl.groupby("Distance").ResponseYesNo.mean()

    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    for ci, c in enumerate(CALLS):
        ax.plot(grid, response_rate(F[ci]), color=colors[c], lw=2.2,
                label=f"{labels[c]} P(回应)")
    ax.axhline(P_REC, color="k", ls="--", lw=1)
    ax.text(4.2, P_REC + 0.01, f"p_rec={P_REC}", fontsize=9)
    ax.axhline(P_BASE, color="gray", ls="--", lw=1)
    ax.text(4.2, P_BASE + 0.01, f"p_base={P_BASE}（地板）", fontsize=9, color="gray")
    ax.axvline(200, color="gray", ls=":", lw=1)
    ax.scatter(rc.index, rc.values, s=60, color="#C00000", marker="o",
               zorder=5, edgecolor="white", label="自然声实测回应率")
    ax.scatter(rc0.index, rc0.values, s=40, color="gray", marker="x",
               zorder=5, label="对照实测回应率")
    ax.set_xscale("log")
    ax.set_xticks(DIST); ax.set_xticklabels([str(int(d)) for d in DIST])
    ax.set_xlabel("传播距离 distance (m)")
    ax.set_ylabel("回应率 P(回应)")
    ax.set_ylim(0, 1)
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    ax.set_title("两阶段回应模型：软阈值 + 地板回应（解释 200m 仍回应）")
    fig.tight_layout()
    fig.savefig(OUT / "two_stage_response.png")
    print(f"\n图已保存：{OUT / 'two_stage_response.png'}")


if __name__ == "__main__":
    main()
