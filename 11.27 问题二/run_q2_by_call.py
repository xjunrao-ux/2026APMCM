"""第二题分型赋权版。运行本文件，结果写入同目录下的 results_by_call。"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gam_baseline.gam_fidelity import (
    CALLS, DIST, METRICS, TAUS, critic_weights, crossing, entropy_weights, setup_font,
)
from gam_baseline.model_comparison import fit_snr, predict_snr
from gam_baseline.two_stage_response import response_rate, sigmoid


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results"
OUT = HERE / "results_by_call"


def main():
    raw = pd.read_csv(INPUT / "propagation_long.csv")
    observed = raw.groupby(["call", "d", "metric"])["correlation"]
    means = observed.mean().unstack("metric")[METRICS]
    # 延续原方案的组内均值标准误。区间条件于当前权重，忽略通道协方差。
    se = observed.sem(ddof=0).unstack("metric")[METRICS]
    gam = pd.read_csv(INPUT / "gam_predictions.csv").query("hold == 0")
    grid = np.sort(gam["d"].unique())
    means_by_call = [means.loc[call].loc[DIST] for call in CALLS]
    snr = predict_snr(fit_snr(np.stack([frame.to_numpy().T for frame in means_by_call])), grid)

    weights, ranges, curves, comparisons, responses = [], [], [], [], []
    thresholds, omissions = [], []
    for ci, (call, frame) in enumerate(zip(CALLS, means_by_call)):
        # 行为七个实测距离，列为三个声学指标。归一化、相关性与权重均在本类型内计算。
        x = frame.to_numpy()
        estimates = gam.loc[gam["call"] == call].pivot(
            index="d", columns="metric", values="estimate"
        ).loc[grid, METRICS].to_numpy()
        if not (np.isfinite(x).all() and np.isfinite(estimates).all()):
            raise ValueError(f"{call} 的距离或通道数据不完整")
        cell_se = se.loc[call].loc[DIST]
        grid_se = np.column_stack([
            np.interp(np.log2(grid), np.log2(DIST), cell_se[metric]) for metric in METRICS
        ])
        for method, w in [("CRITIC", critic_weights(x)), ("entropy", entropy_weights(x))]:
            if not (np.isfinite(w).all() and (w >= 0).all() and np.isclose(w.sum(), 1)):
                raise ValueError(f"{call} 的 {method} 权重无法从这些观测中估计")
            weights.append(dict(call=call, method=method, **dict(zip(METRICS, w))))
            fidelity = estimates @ w
            composite_se = np.sqrt((grid_se**2) @ (w**2))
            snr_fidelity = w @ snr[ci]
            for tau in TAUS:
                distance = crossing(grid, fidelity, tau)
                ranges.append(dict(
                    call=call, method=method, threshold=tau, D=distance,
                    D_lo=crossing(grid, fidelity - composite_se, tau),
                    D_hi=crossing(grid, fidelity + composite_se, tau),
                ))
                comparisons.append(dict(call=call, method=method, threshold=tau,
                                        GAM=distance, SNR=crossing(grid, snr_fidelity, tau)))
            if method != "CRITIC":
                continue
            curves.append(pd.DataFrame(dict(call=call, d=grid, F=fidelity, SE=composite_se)))
            for d in [50, 100, 150, 200]:
                f = np.interp(d, grid, fidelity)
                responses.append(dict(call=call, distance=d, F=f,
                                      P_ident=sigmoid(f), P_response=response_rate(f)))
            for tau in np.linspace(0.05, 0.95, 91):
                thresholds.append(dict(call=call, threshold=tau, D=crossing(grid, fidelity, tau)))

        # 每次只从本类的赋权数据中删一个距离，固定九条原始 GAM 曲线。
        for excluded in DIST:
            w = critic_weights(x[DIST != excluded])
            fidelity = estimates @ w
            for tau in TAUS:
                omissions.append(dict(call=call, omitted_m=excluded, threshold=tau,
                                      **dict(zip(METRICS, w)), D=crossing(grid, fidelity, tau)))

    OUT.mkdir(exist_ok=True)
    curve_frame = pd.concat(curves, ignore_index=True)
    results = {
        "weights_by_call": pd.DataFrame(weights),
        "gam_effective_range": pd.DataFrame(ranges),
        "fidelity_curves": curve_frame,
        "model_weight_comparison": pd.DataFrame(comparisons),
        "two_stage_response": pd.DataFrame(responses),
        "threshold_sweep": pd.DataFrame(thresholds),
        "leave_one_distance_weights": pd.DataFrame(omissions),
    }
    for name, table in results.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")

    setup_font()
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    labels = {"flat": "平直型", "mod": "调制型", "tv": "双声型"}
    for call, part in curve_frame.groupby("call", sort=False):
        line, = ax.plot(part.d, part.F, label=labels[call])
        ax.fill_between(part.d, part.F - part.SE, part.F + part.SE,
                        color=line.get_color(), alpha=0.15)
    for tau in TAUS:
        ax.axhline(tau, ls=":", color="gray", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIST, labels=[str(int(d)) for d in DIST])
    ax.set(xlabel="传播距离（米）", ylabel="声学保真度", ylim=(0, 1),
           title="分型 CRITIC 加权的保真度曲线（阴影为 ±1 标准误）")
    ax.legend(frameon=False)
    fig.savefig(OUT / "gam_fidelity.png", dpi=200)
    plt.close(fig)
    print(results["weights_by_call"].query("method == 'CRITIC'").to_string(index=False))
    print(results["gam_effective_range"].query("method == 'CRITIC'").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
