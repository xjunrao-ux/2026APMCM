"""用题目原始 CSV 核验第二问建模选择。只向终端输出，不改写数据。"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "题目与要求" / "date"
DISTANCE = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ("flat", "mod", "tv")
LENGTHS = np.geomspace(4, 1500, 401)
EXCESS = np.linspace(0, 0.045, 91)


def load_correlations():
    by_metric = []
    for first_file in (3, 6, 9):
        by_call = []
        for file_number in range(first_file, first_file + 3):
            (path,) = DATA.glob(f"Data{file_number:02}_*.csv")
            with path.open(encoding="utf-8-sig") as source:
                header = source.readline().strip().split(",")
                assert header == [f"{d:g}m" for d in DISTANCE], path
                values = np.genfromtxt(source, delimiter=",", missing_values="NA")
            assert values.shape == (300, 7), path
            by_call.append(values)
        by_metric.append(by_call)
    return np.array(by_metric).transpose(1, 0, 2, 3)


def retention(distance, length, excess):
    inverse_ratio = (distance / length) ** 2 * np.exp(excess * (distance - length))
    return 1 / np.sqrt(1 + inverse_ratio)


def fit(means, training, groups, excess_grid):
    # 对每个长度和衰减率，三个近场上限均有有界最小二乘解析解。
    best_error = np.inf
    for excess in excess_grid:
        basis = retention(DISTANCE[None, training], LENGTHS[:, None], excess)
        sum_squares = np.sum(basis**2, axis=1)
        error = 0.0
        parameters = []
        for call in range(len(CALLS)):
            for metrics in groups:
                observed = means[call, metrics][:, training]
                ceilings = np.clip(observed @ basis.T / sum_squares, 0, 1)
                residuals = ceilings.T[:, :, None] * basis[:, None, :] - observed
                errors = np.sum(residuals**2, axis=(1, 2))
                index = np.argmin(errors)
                error += errors[index]
                parameters.append((call, metrics, LENGTHS[index], ceilings[:, index]))
        if error < best_error:
            best_error = error
            best = excess, parameters
    return best


def predict(model, distance):
    excess, parameters = model
    estimates = np.empty((3, 3, len(distance)))
    for call, metrics, length, ceilings in parameters:
        estimates[call, metrics] = ceilings[:, None] * retention(distance, length, excess)
    return estimates


def crossing_ranges(model, weights):
    grid = np.linspace(4, 256, 25201)
    fidelity = np.einsum("m,cmd->cd", weights, predict(model, grid))
    result = []
    for curve in fidelity:
        crossings = []
        for threshold in (0.5, 0.6, 0.7):
            if curve[0] < threshold:
                crossings.append(-np.inf)
            elif curve[-1] >= threshold:
                crossings.append(np.inf)
            else:
                crossings.append(float(np.interp(threshold, curve[::-1], grid[::-1])))
        result.append(crossings)
    return result


def main():
    values = load_correlations()
    valid = np.isfinite(values)
    assert np.all(valid == valid[:, :1])
    assert 0 < values[valid].min() < values[valid].max() < 1
    means = np.nanmean(values, axis=2)
    print("有效相关值数、范围", valid.sum(), values[valid].min(), values[valid].max())
    print("各类型每通道有效数，行序 flat/mod/tv", valid[:, 0].sum(axis=1), sep="\n")
    print("综合保真度观测均值", means.mean(axis=1).round(4), sep="\n")
    print("256 m 三通道均值", means[:, :, -1].round(4), sep="\n")
    delta = means[2, :, -1] - means[1, :, -1]
    spectral_delta = np.mean(delta[:2])
    print("256 m 双声型与调制型排序改变的包络权重", spectral_delta / (spectral_delta - delta[2]))

    equal = np.full(3, 1 / 3)
    training = np.ones(len(DISTANCE), dtype=bool)
    for groups in (([0, 1, 2],), ([0, 1], [2])):
        for excess_grid in ([0.0], EXCESS):
            model = fit(means, training, groups, excess_grid)
            residuals = []
            fold_ranges = []
            for i in range(len(DISTANCE)):
                take = training.copy()
                take[i] = False
                fold = fit(means, take, groups, excess_grid)
                residuals.append(predict(fold, DISTANCE[i:i + 1])[:, :, 0] - means[:, :, i])
                fold_ranges.append(crossing_ranges(fold, equal))
            print("\n通道组", groups, "共用额外衰减率", model[0])
            print("整距离留出 RMSE", np.sqrt(np.mean(np.square(residuals))))
            print("仅留出 256 m 的 RMSE", np.sqrt(np.mean(np.square(residuals[-1]))))
            print("距离，列序 0.5/0.6/0.7", np.round(crossing_ranges(model, equal), 1), sep="\n")
            # 正负无穷是删失标记，分别表示至少 256 m 和低于最近测点。
            limits = np.array(fold_ranges, dtype=float)
            print("删一距离范围下界（-inf 表示 <4 m）", limits.min(axis=0).round(1), sep="\n")
            print("删一距离范围上界（inf 表示 ≥256 m）", limits.max(axis=0).round(1), sep="\n")


if __name__ == "__main__":
    main()
