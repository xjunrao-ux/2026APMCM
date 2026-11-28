# -*- coding: utf-8 -*-
"""
第二问 · 退化模型对比：作者 GAM vs 噪声退化模型(SNR) × 三套权重。

退化模型（拟合）两个：
  1) 作者 GAM：gam(correlation ~ s(d,k=3), REML)，取自 gam_predictions.csv；
  2) 噪声退化模型：μ(d)=a·√(S/(1+S))，S=(L/d)^2，即 μ=a/√(1+(d/L)^2)（球面扩散+平稳噪声）。

权重（合成）三个：CRITIC（主）、熵权法、平均权重。
输出：2 模型 × 3 权重 × 3 叫声 × 3 阈值 的 D(τ)，以及 GAM vs SNR 曲线对比图。
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
TAUS = (0.5, 0.6, 0.7)


def setup_font():
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            fm.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(p)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_means():
    means = np.zeros((3, 3, 7))
    for mi in range(3):
        for ci in range(3):
            (p,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(p)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            means[ci, mi] = df.astype(float).mean(axis=0).to_numpy()
    return means


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


def entropy_weights(X):
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    P = np.clip(Xn / Xn.sum(axis=0), 1e-12, None)
    k = 1.0 / np.log(len(X))
    e = -k * (P * np.log(P)).sum(axis=0)
    return (1 - e) / (1 - e).sum()


def retention(d, L):
    return 1.0 / np.sqrt(1.0 + (d / L) ** 2)


def fit_snr(means):
    L_grid = np.geomspace(4, 1500, 401)
    groups = [(0, 1), (2,)]
    params = []
    for call in range(3):
        for metrics in groups:
            o = means[call, metrics]
            basis = retention(DIST[None, :], L_grid[:, None])
            ss = np.sum(basis ** 2, axis=1)
            ceilings = np.clip(o @ basis.T / ss, 0, 1)
            resid = ceilings.T[:, :, None] * basis[:, None, :] - o
            errs = np.sum(resid ** 2, axis=(1, 2))
            idx = int(np.argmin(errs))
            params.append((call, metrics, float(L_grid[idx]), ceilings[:, idx]))
    return params


def predict_snr(params, d):
    d = np.atleast_1d(d)
    est = np.empty((3, 3, len(d)))
    for call, metrics, L, a in params:
        est[call, metrics] = a[:, None] * retention(d, L)
    return est


def combine(curves, w):
    return np.einsum("m,cmd->cd", w, curves)


def crossing(d, y, tau):
    below = np.flatnonzero(y < tau)
    if not len(below):
        return np.inf
    j = below[0]
    if j == 0:
        return -np.inf
    f = (tau - y[j - 1]) / (y[j] - y[j - 1])
    l, r = d[j - 1], d[j]
    return l * (r / l) ** f


def fmt(x):
    return "≥256" if np.isinf(x) else f"{x:.1f}"


def main():
    setup_font()
    means = load_means()
    grid, gam = load_gam()
    snr = predict_snr(fit_snr(means), grid)

    X = means.transpose(0, 2, 1).reshape(-1, 3)
    w_crit, w_ent, w_eq = critic_weights(X), entropy_weights(X), np.full(3, 1 / 3)
    models = {"作者GAM": gam, "噪声退化(SNR)": snr}
    weights = {"CRITIC(主)": w_crit, "熵权": w_ent, "平均": w_eq}

    # 主结果表
    rows = []
    print("== 有效通信距离 D(τ)：2 退化模型 × 3 权重 ==")
    for mname, curves in models.items():
        print(f"\n【{mname}】")
        print(f"{'权重':<10}{'叫声':<6}{'D(0.5)':>9}{'D(0.6)':>9}{'D(0.7)':>9}")
        for wname, w in weights.items():
            F = combine(curves, w)
            for ci, c in enumerate(CALLS):
                d5 = crossing(grid, F[ci], 0.5)
                d6 = crossing(grid, F[ci], 0.6)
                d7 = crossing(grid, F[ci], 0.7)
                rows.append(dict(model=mname, weight=wname, call=c, D05=d5, D06=d6, D07=d7))
                print(f"{wname:<10}{c:<6}{fmt(d5):>9}{fmt(d6):>9}{fmt(d7):>9}")
    pd.DataFrame(rows).to_csv(OUT / "model_weight_comparison.csv", index=False, encoding="utf-8-sig")

    # 对比图：GAM vs SNR（CRITIC 主权重）
    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200, sharey=True)
    for ax, (mname, curves) in zip(axes, models.items()):
        F = combine(curves, w_crit)
        for ci, c in enumerate(CALLS):
            ax.plot(grid, F[ci], color=colors[c], lw=2.2, label=labels[c])
        for t in (0.5, 0.7):
            ax.axhline(t, color="gray", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(DIST); ax.set_xticklabels([int(d) for d in DIST])
        ax.set_xlabel("距离 (m)"); ax.set_ylim(0, 1); ax.grid(ls=":", alpha=0.4)
        ax.set_title(f"{mname} × CRITIC 主权重")
        if ax is axes[0]:
            ax.set_ylabel("保真度 F")
            ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png")
    print(f"\n图已保存：{OUT / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
