# -*- coding: utf-8 -*-
"""
第二问 · CRITIC 客观赋权保真度 + 信噪比退化曲线
================================================

流程：
  1) 读 Data03–11，计算 63 个「叫声 × 通道 × 距离」单元均值；
  2) CRITIC 客观赋权：以 21 个「叫声 × 距离」单元为样本、三条通道为指标，
     按「对比强度(标准差) × 冲突度(1-相关系数)」确定三通道权重 w=(ws,wa,we)；
  3) 用 w 计算三种叫声的 F：加权算术平均 与 加权几何平均，并比对；
  4) 用信噪比导出的物理退化曲线 μ(d)=a/√(1+(d/L)^2) 逐通道拟合，
     声谱图与幅度谱共用一个距离尺度 L，包络单独一个 L；
  5) 把三条 SNR 通道按 CRITIC 权重合成为 F(d)，画三种叫声的退化曲线
     （信噪比平滑曲线，并叠加三次样条插值作对比）。

依赖：numpy, pandas, matplotlib。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline


def setup_cjk_font():
    """挂一个 Windows 中文字体，避免图中中文变方框。找不到则返回 False。"""
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            fm.fontManager.addfont(str(p))
            prop = fm.FontProperties(fname=str(p))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

# ---------- 配置 ----------
DATA_DIR = Path(r"C:\Users\34232\Documents\GitHub\2026APMCM\题目与要求\date")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ["flat", "mod", "tv"]
METRICS = ["spectrogram", "spectrum", "envelope"]

# 文件名映射：Data{03..11}
# 通道顺序：声谱图(spectrogram) 3/4/5，幅度谱(spectrum) 6/7/8，包络(envelope) 9/10/11
FILE_GRID = np.array([
    [3, 4, 5],   # spectrogram: flat, mod, tv
    [6, 7, 8],   # spectrum
    [9, 10, 11], # envelope
])


def load_means():
    """读 9 个 CSV，返回 shape=(3叫声, 3通道, 7距离) 的单元均值。"""
    means = np.zeros((3, 3, 7))
    for mi in range(3):
        for ci in range(3):
            (path,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(path)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            means[ci, mi] = df.astype(float).mean(axis=0).to_numpy()
    return means


# ============ CRITIC 客观赋权 ============
def critic_weights(X):
    """
    X: (n样本, n指标) 决策矩阵，这里 n=21(叫声×距离), n指标=3(通道)。
    返回 (weights, 对比强度σ, 冲突度, 信息量C)。
    """
    # 1) 正向指标 min-max 归一化到 [0,1]（三种相关性都是越大越保真）
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # 2) 对比强度 = 各指标标准差
    sigma = Xn.std(axis=0, ddof=0)
    # 3) 冲突度 = 与其它指标的 (1 - Pearson r) 之和（对角 (1-r_ii)=0 已自动排除自身）
    r = np.corrcoef(Xn, rowvar=False)
    conflict = (1 - r).sum(axis=1)
    # 4) 信息量 C_j = σ_j × 冲突度_j
    C = sigma * conflict
    w = C / C.sum()
    return w, sigma, conflict, C


def fidelity_arith(channels, w):
    """加权算术平均。channels: (n_call, 3通道, n_dist)，第 1 维是通道。"""
    return np.einsum("m,cmd->cd", w, channels)


def fidelity_geo(channels, w):
    """加权几何平均。channels 各分量 >0。"""
    c = np.clip(channels, 1e-12, None)
    return np.exp(np.einsum("m,cmd->cd", w, np.log(c)))


# ============ 信噪比退化模型 ============
def retention(d, L):
    """μ(d)=a/√(1+(d/L)^2)。由球面扩散(功率∝1/d²)+平稳噪声，相关性随信噪比 √(S/(1+S)) 而来。"""
    return 1.0 / np.sqrt(1.0 + (d / L) ** 2)


def fit_snr(means):
    """
    每类叫声拟合 SNR 模型：声谱图+幅度谱共用一个 L，包络单独一个 L。
    对 L 网格搜索，近场上限 a 用有界最小二乘闭式解。
    返回 parameters = [(call, metrics_group, L, a_vector), ...]
    """
    L_grid = np.geomspace(4, 1500, 401)
    groups = [(0, 1), (2,)]  # 频谱组, 包络组
    parameters = []
    for call in range(3):
        for metrics in groups:
            observed = means[call, metrics]          # (n_metrics, 7)
            basis = retention(DIST[None, :], L_grid[:, None])  # (n_L, 7)
            ss = np.sum(basis ** 2, axis=1)          # (n_L,)
            ceilings = np.clip(observed @ basis.T / ss, 0, 1)  # (n_metrics, n_L)
            resid = ceilings.T[:, :, None] * basis[:, None, :] - observed
            errors = np.sum(resid ** 2, axis=(1, 2))  # (n_L,)
            idx = int(np.argmin(errors))
            parameters.append((call, metrics, float(L_grid[idx]), ceilings[:, idx]))
    return parameters


def predict_snr(parameters, d):
    """在距离 d（标量或数组）上给出三条通道的 SNR 预测，shape=(3叫声,3通道,len(d))。"""
    est = np.empty((3, 3, len(np.atleast_1d(d))))
    for call, metrics, L, a in parameters:
        est[call, metrics] = a[:, None] * retention(np.atleast_1d(d), L)
    return est


def main():
    setup_cjk_font()
    means = load_means()  # (3,3,7)

    # ---- CRITIC：21 个「叫声×距离」样本 × 3 通道 ----
    X = means.transpose(0, 2, 1).reshape(-1, 3)  # (21,3)
    w, sigma, conflict, C = critic_weights(X)

    crit = pd.DataFrame({
        "metric": METRICS,
        "weight": w,
        "contrast_sigma": sigma,
        "conflict": conflict,
        "information_C": C,
    })
    crit.to_csv(OUT / "critic_weights.csv", index=False, encoding="utf-8-sig")
    print("== CRITIC 权重 ==")
    print(crit.round(4).to_string(index=False))

    # ---- 用 CRITIC 权重算 7 个实测距离上的 F（原始单元均值）----
    f_ari = fidelity_arith(means, w)   # (3,7)
    f_geo = fidelity_geo(means, w)     # (3,7)
    rows = []
    for ci, c in enumerate(CALLS):
        for k, d in enumerate(DIST):
            rows.append(dict(call=c, distance=int(d),
                             F_arith_critic=f_ari[ci, k], F_geo_critic=f_geo[ci, k]))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "fidelity_critic_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n== 三种叫声的 F（CRITIC 加权）== 前几行 ==")
    print(comp.head(7).round(4).to_string(index=False))

    # ---- 拟合 SNR 退化模型 ----
    params = fit_snr(means)
    snr_rows = []
    for call, metrics, L, a in params:
        for j, mi in enumerate(metrics):
            snr_rows.append(dict(call=CALLS[call], metric=METRICS[mi],
                                 length_L_m=L, ceiling_a=a[j]))
    snr = pd.DataFrame(snr_rows)
    snr.to_csv(OUT / "snr_parameters.csv", index=False, encoding="utf-8-sig")
    print("\n== SNR 模型参数 ==")
    print(snr.round(4).to_string(index=False))

    # ---- 在细网格上合成 F(d) 并画图 ----
    grid = np.geomspace(4, 256, 400)
    est = predict_snr(params, grid)     # (3,3,400) SNR 模型
    F_ari = fidelity_arith(est, w)      # (3,400) 信噪比退化曲线（算术）
    F_geo = fidelity_geo(est, w)        # (3,400) 信噪比退化曲线（几何，仅存表，不画）

    # 三次样条插值：对 7 个实测 CRITIC 加权算术 F 做三次样条，用于对比
    spline_curves = np.empty_like(F_ari)
    for ci in range(3):
        spline_curves[ci] = CubicSpline(DIST, f_ari[ci])(grid)

    # 保存 SNR vs 样条 曲线值
    spline_df = pd.DataFrame({"distance": np.round(grid, 4)})
    for ci, c in enumerate(CALLS):
        spline_df[f"F_snr_{c}"] = F_ari[ci]
        spline_df[f"F_spline_{c}"] = spline_curves[ci]
    spline_df.to_csv(OUT / "spline_vs_snr.csv", index=False, encoding="utf-8-sig")

    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    for ci, c in enumerate(CALLS):
        ax.plot(grid, F_ari[ci], color=colors[c], lw=2.4,
                label=f"{labels[c]} 信噪比模型")
        ax.plot(grid, spline_curves[ci], color=colors[c], lw=1.5, ls="--",
                label=f"{labels[c]} 三次样条")
        ax.scatter(DIST, f_ari[ci], color=colors[c], s=26, zorder=5, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xticks(DIST)
    ax.set_xticklabels([str(int(d)) for d in DIST])
    ax.set_xlabel("传播距离 distance (m)")
    ax.set_ylabel("声保真度 F（CRITIC 加权算术）")
    ax.set_ylim(0, 1)
    ax.set_title("三种叫声的退化曲线：信噪比模型 vs 三次样条插值")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "degradation_curves.png")
    fig.savefig(OUT / "degradation_curves.pdf")
    print(f"\n图已保存：{OUT / 'degradation_curves.png'}")

    # 报告三次样条越界（过冲/欠冲）情况
    print("\n== 三次样条越界检查（F 应在 [0,1]，样条可能过冲）==")
    for ci, c in enumerate(CALLS):
        print(f"  {c:<5} 样条 min={spline_curves[ci].min():.3f}  max={spline_curves[ci].max():.3f}")

    print("\n== 有效通信距离 D(T)（SNR 曲线 × CRITIC 权重，首次穿阈，对数插值）==")
    for ci, c in enumerate(CALLS):
        for T in (0.5, 0.6, 0.7):
            below = np.flatnonzero(F_ari[ci] < T)
            if not len(below):
                print(f"  {c:<5} D({T}) ≥ 256 m（未跌破）")
            elif below[0] == 0:
                print(f"  {c:<5} D({T}) < 4 m")
            else:
                j = below[0]
                f = (T - F_ari[ci, j - 1]) / (F_ari[ci, j] - F_ari[ci, j - 1])
                left, right = grid[j - 1], grid[j]
                D = left * (right / left) ** f
                print(f"  {c:<5} D({T}) = {D:6.1f} m")


if __name__ == "__main__":
    main()
