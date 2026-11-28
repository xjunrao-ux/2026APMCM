# -*- coding: utf-8 -*-
"""
第二问后期：有效通信距离估计 + 不确定度范围 + 敏感性分析 + 稳健性比较。

只使用 Data03–11（传播相关性），不使用 Data02。

主流程：
  1) 读 9 张相关性表，得到 63 个「叫声×通道×距离」单元均值；
  2) CRITIC 客观赋权确定包络权重 w_env（频谱两条平分 1-w_env）；
  3) 信噪比退化模型 μ(d)=a/√(1+(d/L)²) 逐通道拟合（频谱组共享 L，包络单独 L）；
  4) 合成 F(d)，在 τ=0.5(潜在识别) 与 τ=0.7(可靠识别) 下首次穿阈求有效距离 D(τ)；
  5) 不确定度：删一距离重拟合；缺缺失压力测试（缺失→Q25 / 缺失→0）；
  6) 敏感性：阈值扫描 τ∈(0,1)；包络权重扫描 w_env∈[0,1]；
  7) 稳健性比较与结论。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA_DIR = Path(r"C:\Users\34232\Documents\GitHub\2026APMCM\题目与要求\date")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ["flat", "mod", "tv"]
METRICS = ["spectrogram", "spectrum", "envelope"]
FILE_GRID = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
GRID = np.geomspace(4, 256, 2000)
TAUS = (0.5, 0.6, 0.7)


def setup_font():
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            fm.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(p)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_raw():
    """返回 (3叫声,3通道,300样本,7距离) 原始相关值（含 NaN）。"""
    raw = np.full((3, 3, 300, 7), np.nan)
    for mi in range(3):
        for ci in range(3):
            (path,) = DATA_DIR.glob(f"Data{FILE_GRID[mi, ci]:02d}_*.csv")
            df = pd.read_csv(path)
            df.columns = [int(float(str(c).replace("m", ""))) for c in df.columns]
            raw[ci, mi] = df.astype(float).to_numpy()
    return raw


def critic_weights(X):
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    sigma = Xn.std(axis=0, ddof=0)
    r = np.corrcoef(Xn, rowvar=False)
    conflict = (1 - r).sum(axis=1)
    C = sigma * conflict
    return C / C.sum()


def retention(d, L):
    return 1.0 / np.sqrt(1.0 + (d / L) ** 2)


def fit_snr(means, keep):
    """信噪比模型拟合。means:(3,3,7)，keep: 长度为7的布尔掩码。"""
    L_grid = np.geomspace(4, 1500, 401)
    groups = [(0, 1), (2,)]
    d = DIST[keep]
    obs = means[:, :, keep]
    params = []
    for call in range(3):
        for metrics in groups:
            o = obs[call, metrics]
            basis = retention(d[None, :], L_grid[:, None])
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


def fidelity(channels, w_env):
    """包络权重 w_env，频谱两条平分 1-w_env。channels:(3,3,N)。"""
    w = np.array([(1 - w_env) / 2, (1 - w_env) / 2, w_env])
    return np.einsum("m,cmd->cd", w, channels)


def crossing(d, y, tau):
    below = np.flatnonzero(y < tau)
    if not len(below):
        return np.inf                    # 未跌破 → ≥256
    j = below[0]
    if j == 0:
        return -np.inf                   # 4m 即低于阈值
    f = (tau - y[j - 1]) / (y[j] - y[j - 1])
    l, r = d[j - 1], d[j]
    return l * (r / l) ** f


def D_of(params, w_env, tau):
    F = fidelity(predict_snr(params, GRID), w_env)
    return np.array([crossing(GRID, F[ci], tau) for ci in range(3)])


def fmt(x):
    if np.isinf(x):
        return "≥256" if x > 0 else "<4"
    return f"{x:.1f}"


def main():
    setup_font()
    raw = load_raw()
    means = np.nanmean(raw, axis=2)                      # (3,3,7)
    X = means.transpose(0, 2, 1).reshape(-1, 3)          # (21,3)
    w_crit = critic_weights(X)
    w_env = w_crit[2]
    print(f"CRITIC 权重：声谱图 {w_crit[0]:.4f} / 幅度谱 {w_crit[1]:.4f} / 包络 {w_crit[2]:.4f}")

    params_full = fit_snr(means, np.ones(7, dtype=bool))

    # ---------- 1) 主结果：两条判据下的有效距离 ----------
    print("\n== 主结果：有效通信距离 D(τ)（信噪比模型 × CRITIC 权重）==")
    D_full = {t: D_of(params_full, w_env, t) for t in TAUS}
    print(f"{'call':<6}{'D(0.5)潜在':>12}{'D(0.6)':>10}{'D(0.7)可靠':>12}")
    for ci, c in enumerate(CALLS):
        print(f"{c:<6}{D_full[0.5][ci]:>10.1f} m{D_full[0.6][ci]:>8.1f} m{D_full[0.7][ci]:>10.1f} m")

    # ---------- 2) 删一距离不确定度 ----------
    loo_rows, range_rows = [], []
    for omit in range(7):
        keep = np.ones(7, dtype=bool); keep[omit] = False
        p = fit_snr(means, keep)
        for t in (0.5, 0.7):
            d = D_of(p, w_env, t)
            for ci, c in enumerate(CALLS):
                loo_rows.append(dict(omitted_m=int(DIST[omit]), threshold=t, call=c, D=d[ci]))
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT / "leave_one_out.csv", index=False, encoding="utf-8-sig")
    print("\n== 删一距离不确定度范围 ==")
    print(f"{'call':<6}{'判据':<6}{'全数据D':>10}{'删距min':>10}{'删距max':>10}{'删失折数':>8}")
    for ci, c in enumerate(CALLS):
        for t in (0.5, 0.7):
            sub = loo[(loo.call == c) & (loo.threshold == t)].D
            fin = sub[np.isfinite(sub)]
            cen = int((~np.isfinite(sub)).sum())
            full = D_full[t][ci]
            range_rows.append(dict(call=c, threshold=t, D_full=full,
                                   loo_min=fin.min(), loo_max=fin.max(), censored=c))
            print(f"{c:<6}{t:<6}{full:>8.1f} m{fin.min():>8.1f} m{fin.max():>8.1f} m{cen:>8}")
    pd.DataFrame(range_rows).to_csv(OUT / "effective_range.csv", index=False, encoding="utf-8-sig")

    # ---------- 3) 缺缺失压力测试 ----------
    print("\n== 缺缺失压力测试（缺失→Q25 / 缺失→0）D(τ) ==")
    q25 = np.nanquantile(raw, 0.25, axis=2)              # (3,3,7)
    scen = {
        "available": means,
        "missing_at_q25": np.where(np.isnan(raw), q25[:, :, None, :], raw).mean(axis=2),
        "missing_at_zero": np.where(np.isnan(raw), 0.0, raw).mean(axis=2),
    }
    miss_rows = []
    for name, m in scen.items():
        p = fit_snr(m, np.ones(7, dtype=bool))
        for t in (0.5, 0.7):
            d = D_of(p, w_env, t)
            for ci, c in enumerate(CALLS):
                miss_rows.append(dict(scenario=name, threshold=t, call=c, D=d[ci]))
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(OUT / "missingness_sensitivity.csv", index=False, encoding="utf-8-sig")
    for name in ("available", "missing_at_q25", "missing_at_zero"):
        sub = miss[miss.scenario == name]
        line = f"  {name:<16}"
        for t in (0.5, 0.7):
            d = {r.call: r.D for r in sub[sub.threshold == t].itertuples()}
            line += f"  {t}: " + "/".join(f"{c}={fmt(d[c])}" for c in CALLS)
        print(line)

    # ---------- 4) 阈值扫描 ----------
    tau_grid = np.linspace(0.05, 0.95, 91)
    F_full = fidelity(predict_snr(params_full, GRID), w_env)
    sweep = []
    for t in tau_grid:
        for ci, c in enumerate(CALLS):
            sweep.append(dict(threshold=t, call=c, D=crossing(GRID, F_full[ci], t)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False, encoding="utf-8-sig")

    # ---------- 5) 权重扫描 ----------
    w_grid = np.linspace(0, 1, 101)
    wsweep = []
    for w in w_grid:
        for t in (0.5, 0.7):
            d = D_of(params_full, w, t)
            for ci, c in enumerate(CALLS):
                wsweep.append(dict(envelope_weight=w, threshold=t, call=c, D=d[ci]))
    wsweep = pd.DataFrame(wsweep)
    wsweep.to_csv(OUT / "weight_sweep.csv", index=False, encoding="utf-8-sig")

    # 找 tv/mod 在 0.5、0.7 下的排序翻转点
    print("\n== 权重扫描：tv 与 mod 的排序翻转点 ==")
    for t in (0.5, 0.7):
        sub = wsweep[wsweep.threshold == t].pivot(index="envelope_weight", columns="call", values="D")
        diff = sub["tv"] - sub["mod"]
        flip = w_grid[np.where(np.diff(np.signbit(diff)))[0]]
        if len(flip):
            print(f"  τ={t}: tv-mod 差符号翻转于包络权重 w≈{flip[0]:.3f}")
        else:
            print(f"  τ={t}: 0–1 范围内 tv/mod 排序不变")

    # ---------- 6) 稳健性：各距离 F 与远距离排序 ----------
    F_at_dist = fidelity(predict_snr(params_full, DIST), w_env)  # (3,7)
    print("\n== 稳健性：各距离 CRITIC 加权 F ==")
    print(f"{'distance':<10}" + "".join(f"{c:>10}" for c in CALLS))
    for k, dd in enumerate(DIST):
        print(f"{int(dd):<10}" + "".join(f"{F_at_dist[ci, k]:>10.3f}" for ci in range(3)))
    for dd, k in [(128, 5), (256, 6)]:
        order = [CALLS[i] for i in np.argsort(-F_at_dist[:, k])]
        print(f"  {dd} m 排序: {' > '.join(order)}")

    # ---------- 图 ----------
    colors = {"flat": "#7CCD7C", "mod": "#8B4726", "tv": "#27408B"}
    labels = {"flat": "平直型 flat", "mod": "调制型 mod", "tv": "双声型 tv"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=200)
    for ci, c in enumerate(CALLS):
        s = sweep[sweep.call == c]
        d = s.D.replace([np.inf, -np.inf], [256, 4]).to_numpy()
        a1.plot(s.threshold, d, color=colors[c], lw=2, label=labels[c])
    a1.set_xlabel("识别阈值 τ")
    a1.set_ylabel("有效通信距离 D (m)")
    a1.set_title("阈值敏感性（CRITIC 权重）")
    a1.grid(ls=":", alpha=0.4); a1.legend(frameon=False)

    for t, ls in [(0.5, "-"), (0.7, "--")]:
        for ci, c in enumerate(CALLS):
            s = wsweep[(wsweep.threshold == t) & (wsweep.call == c)].sort_values("envelope_weight")
            d = s.D.replace([np.inf, -np.inf], [256, 4]).to_numpy()
            a2.plot(s.envelope_weight, d, color=colors[c], lw=2, ls=ls,
                    label=f"{labels[c]} τ={t}" if t == 0.5 else None)
    a2.axvline(w_env, color="k", ls=":", lw=1)
    a2.text(w_env, 245, f"CRITIC w={w_env:.2f}", rotation=90, fontsize=8, va="top")
    a2.axvline(1 / 3, color="gray", ls=":", lw=1)
    a2.text(1 / 3, 245, "等权 1/3", rotation=90, fontsize=8, va="top")
    a2.set_xlabel("包络权重 w（频谱平分 1−w）")
    a2.set_ylabel("有效通信距离 D (m)")
    a2.set_title("权重敏感性")
    a2.grid(ls=":", alpha=0.4); a2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png")
    print(f"\n图已保存：{OUT / 'sensitivity.png'}")


if __name__ == "__main__":
    main()
