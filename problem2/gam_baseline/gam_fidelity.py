# -*- coding: utf-8 -*-
"""
第二问（以作者 GAM 为基线）：多指标保真度 + 分级阈值 + 有效通信距离。

基线 = 作者 Code2 的 GAM（gam(correlation ~ s(d, k=3), method="REML")），
曲线取自团队已运行的 gam_predictions.csv（hold=0 为全数据 GAM）。

流程：
  1) 读 hold=0 的 GAM 曲线（3 叫声 × 3 通道，4–256m 共 2521 点）；
  2) 读原始 Data03–11 得各「叫声×通道×距离」均值/标准差/有效数（用于 CRITIC 与不确定度）；
  3) 多指标保真度：把三条 GAM 通道加权合成 F(d)（CRITIC 为主，熵权/等权/作者 GAM 为对照）；
  4) 分级阈值 0.5/0.6/0.7，首次穿阈求有效通信距离 D(τ)；
  5) 不确定度：SE=SD/√N 的 ±1 标准误带，反解 D(τ) 范围；
  6) 敏感性：阈值扫描 + 包络权重扫描；
  7) 与作者 Supplementary 4 的 GAM 90% 置信带对照说明。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "题目与要求" / "date"
GAM_CSV = HERE.parent / "results" / "gam_predictions.csv"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

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


def load_gam():
    """返回 (grid, gam)，gam 形状 (3叫声,3通道,2521)，通道序 = spectrogram/spectrum/envelope。"""
    df = pd.read_csv(GAM_CSV)
    full = df[df.hold == 0]
    grid = np.sort(full.d.unique())
    gam = np.empty((3, 3, len(grid)))
    for ci, c in enumerate(CALLS):
        for mi, m in enumerate(METRICS):
            sub = full[(full.call == c) & (full.metric == m)].sort_values("d")
            gam[ci, mi] = sub.estimate.to_numpy()
    return grid, gam


def load_stats():
    """返回 means/SD/N，各为 (3叫声,3通道,7距离)。"""
    means = np.zeros((3, 3, 7)); sd = np.zeros((3, 3, 7)); n = np.zeros((3, 3, 7))
    for mi in range(3):
        for ci in range(3):
            (p,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(p)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            a = df.astype(float).to_numpy()
            means[ci, mi] = np.nanmean(a, axis=0)
            sd[ci, mi] = np.nanstd(a, axis=0)
            n[ci, mi] = np.sum(np.isfinite(a), axis=0)
    return means, sd, n


def critic_weights(X):
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    sigma = Xn.std(axis=0, ddof=0)
    r = np.corrcoef(Xn, rowvar=False)
    conflict = (1 - r).sum(axis=1)
    C = sigma * conflict
    return C / C.sum()


def entropy_weights(X):
    """熵权法：指标值越分散(熵越小)→信息量越大→权重越大。"""
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    P = np.clip(Xn / Xn.sum(axis=0), 1e-12, None)
    k = 1.0 / np.log(len(X))
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def combine(gam, w):
    """gam (3,3,N)，w (3,) 通道权重 → F (3,N)。"""
    return np.einsum("m,cmd->cd", w, gam)


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


def distance_bounds(grid, fidelity, se, threshold):
    """较低的保真度曲线先跌破阈值，给出距离下界。"""
    lower = crossing(grid, fidelity - se, threshold)
    upper = crossing(grid, fidelity + se, threshold)
    return lower, upper


def main():
    setup_font()
    grid, gam = load_gam()
    means, sd, n = load_stats()

    # 三套权重（用 21 个单元均值）：CRITIC 为主，熵权法 / 平均权重 / 作者 GAM 为对照
    X = means.transpose(0, 2, 1).reshape(-1, 3)
    w_crit = critic_weights(X)
    w_ent = entropy_weights(X)
    w_eq = np.full(3, 1 / 3)
    weights = {"critic(主)": w_crit, "entropy": w_ent, "equal": w_eq}
    print("三套权重（声谱图 / 幅度谱 / 包络）：")
    print(f"  CRITIC(主) {w_crit[0]:.3f} / {w_crit[1]:.3f} / {w_crit[2]:.3f}")
    print(f"  熵权法     {w_ent[0]:.3f} / {w_ent[1]:.3f} / {w_ent[2]:.3f}")
    print(f"  平均权重   {w_eq[0]:.3f} / {w_eq[1]:.3f} / {w_eq[2]:.3f}")
    pd.DataFrame({"metric": METRICS, "critic": w_crit, "entropy": w_ent, "equal": w_eq}).to_csv(
        OUT / "weights_comparison.csv", index=False, encoding="utf-8-sig")

    # 标准误 SE=SD/sqrt(N)，插值到 GAM 网格（对数距离线性）
    se = sd / np.sqrt(np.maximum(n, 1))          # (3,3,7)
    se_grid = np.zeros((3, 3, len(grid)))
    for ci in range(3):
        for mi in range(3):
            se_grid[ci, mi] = np.interp(np.log2(grid), np.log2(DIST), se[ci, mi])

    # 主结果表
    print("\n== 有效通信距离 D(τ)（作者 GAM 基线）==")
    for wname, w in weights.items():
        F = combine(gam, w)
        SE_F = np.sqrt(np.einsum("m,cmd->cd", w ** 2, se_grid ** 2))
        print(f"\n【{wname} 权重】")
        print(f"{'call':<6}{'D(0.5)':>10}{'D(0.6)':>10}{'D(0.7)':>10}{'D(0.5)范围':>16}{'D(0.7)范围':>16}")
        for ci, c in enumerate(CALLS):
            d5, d7 = crossing(grid, F[ci], 0.5), crossing(grid, F[ci], 0.7)
            d5_lo, d5_hi = distance_bounds(grid, F[ci], SE_F[ci], 0.5)
            d7_lo, d7_hi = distance_bounds(grid, F[ci], SE_F[ci], 0.7)
            def f(x):
                if np.isneginf(x):
                    return "<4"
                return "≥256" if np.isinf(x) else f"{x:.1f}"
            print(f"{c:<6}{d5:>8.1f}{crossing(grid, F[ci], 0.6):>8.1f}{d7:>8.1f}"
                  f"{f'({f(d5_lo)}–{f(d5_hi)})':>16}{f'({f(d7_lo)}–{f(d7_hi)})':>16}")

    # 保存主结果（三套权重）
    rows = []
    for wname, w in weights.items():
        F = combine(gam, w)
        SE_F = np.sqrt(np.einsum("m,cmd->cd", w ** 2, se_grid ** 2))
        for ci, c in enumerate(CALLS):
            for t in TAUS:
                d = crossing(grid, F[ci], t)
                d_lo, d_hi = distance_bounds(grid, F[ci], SE_F[ci], t)
                rows.append(dict(weight=wname, call=c, threshold=t, D=d, D_lo=d_lo, D_hi=d_hi))
    pd.DataFrame(rows).to_csv(OUT / "gam_effective_range.csv", index=False, encoding="utf-8-sig")

    # 主权重（CRITIC）用于画图与敏感性
    F_main = combine(gam, w_crit)
    SE_F_main = np.sqrt(np.einsum("m,cmd->cd", w_crit ** 2, se_grid ** 2))

    # 阈值扫描 + 权重扫描
    tau_grid = np.linspace(0.05, 0.95, 91)
    w_grid = np.linspace(0, 1, 101)
    sweep, wsweep = [], []
    for t in tau_grid:
        for ci, c in enumerate(CALLS):
            sweep.append(dict(threshold=t, call=c, D=crossing(grid, F_main[ci], t)))
    for w in w_grid:
        wv = np.array([(1 - w) / 2, (1 - w) / 2, w])
        Fw = combine(gam, wv)
        for t in (0.5, 0.7):
            for ci, c in enumerate(CALLS):
                wsweep.append(dict(envelope_weight=w, threshold=t, call=c,
                                   D=crossing(grid, Fw[ci], t)))
    sweep = pd.DataFrame(sweep); wsweep = pd.DataFrame(wsweep)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    wsweep.to_csv(OUT / "weight_sweep.csv", index=False, encoding="utf-8-sig")

    # 排序翻转点
    print("\n== 权重扫描：tv/mod 排序翻转点 ==")
    for t in (0.5, 0.7):
        sub = wsweep[wsweep.threshold == t].pivot(index="envelope_weight", columns="call", values="D")
        diff = sub["tv"] - sub["mod"]
        flip = w_grid[np.where(np.diff(np.signbit(diff)))[0]]
        print(f"  τ={t}: " + (f"w≈{flip[0]:.3f}" if len(flip) else "0–1 内不变"))

    # 图
    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=200)
    # 左：GAM 三通道 + 保真度
    for ci, c in enumerate(CALLS):
        a1.plot(grid, F_main[ci], color=colors[c], lw=2.4, label=labels[c])
        a1.fill_between(grid, F_main[ci] - SE_F_main[ci], F_main[ci] + SE_F_main[ci],
                        color=colors[c], alpha=0.15)
    for t in (0.5, 0.7):
        a1.axhline(t, color="gray", ls=":", lw=1)
    a1.set_xscale("log"); a1.set_xticks(DIST); a1.set_xticklabels([int(d) for d in DIST])
    a1.set_xlabel("距离 (m)"); a1.set_ylabel("保真度 F（CRITIC）")
    a1.set_title("GAM 基线保真度（CRITIC 主权重，±1SE）"); a1.set_ylim(0, 1); a1.grid(ls=":", alpha=0.4); a1.legend(frameon=False)
    # 中：阈值扫描
    for ci, c in enumerate(CALLS):
        s = sweep[sweep.call == c]
        d = s.D.replace([np.inf, -np.inf], [256, 4]).to_numpy()
        a2.plot(s.threshold, d, color=colors[c], lw=2, label=labels[c])
    a2.set_xlabel("阈值 τ"); a2.set_ylabel("D (m)"); a2.set_title("阈值敏感性")
    a2.grid(ls=":", alpha=0.4); a2.legend(frameon=False)
    # 右：权重扫描
    for t, ls in [(0.5, "-"), (0.7, "--")]:
        for ci, c in enumerate(CALLS):
            s = wsweep[(wsweep.threshold == t) & (wsweep.call == c)].sort_values("envelope_weight")
            d = s.D.replace([np.inf, -np.inf], [256, 4]).to_numpy()
            a3.plot(s.envelope_weight, d, color=colors[c], lw=2, ls=ls,
                    label=f"{labels[c]} τ={t}" if t == 0.5 else None)
    a3.axvline(1 / 3, color="gray", ls=":", lw=1); a3.text(1 / 3, 245, "等权", fontsize=8)
    a3.set_xlabel("包络权重 w"); a3.set_ylabel("D (m)"); a3.set_title("权重敏感性")
    a3.grid(ls=":", alpha=0.4); a3.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "gam_fidelity.png")
    print(f"\n图已保存：{OUT / 'gam_fidelity.png'}")


if __name__ == "__main__":
    main()
