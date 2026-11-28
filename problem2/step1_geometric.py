# -*- coding: utf-8 -*-
"""
计划第1步：几何平均保真度对照（唯一变量 = 三通道合成方式）。
方法完全复刻团队主模型：每通道做单调不升保序拟合 -> 合成 -> 对数距离插值 -> 首次穿阈。
先用自己的 PAVA 保序实现核对团队的 scipy isotonic 拟合值，再算几何平均。
"""
import numpy as np
import pandas as pd

# 我的通道观测均值（已与团队逐位核对一致）
MINE = r"G:\亚太数模\2026APMCM\problem2\results\fidelity_by_distance.csv"
# 团队的拟合结果（用于校验）
THEIR_CH = r"C:\Users\34232\Documents\GitHub\2026APMCM\problem2\results\channel_curves.csv"
THEIR_FK = r"C:\Users\34232\Documents\GitHub\2026APMCM\problem2\results\fidelity_knots.csv"

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
CALLS = ["flat", "mod", "tv"]
LEVELS = [0.5, 0.6, 0.7]
METRICS = ["r_spectrogram", "r_amp_spectrum", "r_envelope"]


def pava_noninc(y):
    """池相邻违例者(PAVA)，单调不升保序回归。等价 scipy.isotonic_regression(..., increasing=False)。"""
    y = np.asarray(y, dtype=float)
    means = list(y)
    sizes = [1] * len(y)
    i = 0
    while i < len(means) - 1:
        if means[i] < means[i + 1]:          # 违例：应不升却上升
            n = sizes[i] + sizes[i + 1]
            m = (means[i] * sizes[i] + means[i + 1] * sizes[i + 1]) / n
            means[i], sizes[i] = m, n
            del means[i + 1]; del sizes[i + 1]
            i = max(0, i - 1)                 # 回退检查前一合并块
        else:
            i += 1
    return np.repeat(means, sizes)


def crossing(x, y, th):
    """首次跌破阈值的距离，对数距离几何插值（复刻团队 crossing()）。"""
    below = np.flatnonzero(y < th)
    if not len(below):
        return np.inf                       # right censored
    j = below[0]
    if j == 0:
        return -np.inf                      # left censored
    f = (th - y[j - 1]) / (y[j] - y[j - 1])
    left, right = x[j - 1], x[j]
    return left * (right / left) ** f


# ---- 1) 取我的通道观测均值，做保序拟合 ----
mine = pd.read_csv(MINE)
obs = np.zeros((3, 3, 7))
for ci, c in enumerate(CALLS):
    sub = mine[mine.call_type == c].sort_values("distance")
    for mi, m in enumerate(METRICS):
        obs[ci, mi] = sub[m].values
fits = np.array([[pava_noninc(obs[ci, mi]) for mi in range(3)] for ci in range(3)])

# ---- 2) 校验：我的 PAVA vs 团队 scipy isotonic ----
their = pd.read_csv(THEIR_CH)
maxdiff = 0.0
for ci, c in enumerate(CALLS):
    for mi, m in enumerate(["spectrogram", "spectrum", "envelope"]):
        tv = their[(their.call == c) & (their.metric == m)].sort_values("d").fitted.values
        maxdiff = max(maxdiff, float(np.max(np.abs(fits[ci, mi] - tv))))
print(f"PAVA 保序拟合 vs 团队 scipy isotonic 最大绝对差 = {maxdiff:.2e}  (应≈0)")

# ---- 3) 合成：算术(复刻) 与 几何 ----
ari = fits.mean(axis=1)                       # (3,7) 等权算术
geo = np.exp(np.mean(np.log(fits), axis=1))   # (3,7) 等权几何

# 校验算术合成与团队 fidelity_knots 一致
their_fk = pd.read_csv(THEIR_FK)
maxdiff_ari = 0.0
for ci, c in enumerate(CALLS):
    tv = their_fk[their_fk.call == c].sort_values("d").fitted.values
    maxdiff_ari = max(maxdiff_ari, float(np.max(np.abs(ari[ci] - tv))))
print(f"算术合成 vs 团队 fidelity_knots 最大绝对差 = {maxdiff_ari:.2e}  (应≈0)")

# ---- 4) 输出两种合成的保真度曲线 + 穿阈距离 ----
print("\n" + "=" * 88)
print("等权算术 vs 等权几何：7 个距离上的保真度（保序拟合后合成）")
print("=" * 88)
for ci, c in enumerate(CALLS):
    print(f"\n【{c}】")
    for i, d in enumerate(DIST):
        print(f"  {d:>4} m   算术 {ari[ci, i]:.4f}   几何 {geo[ci, i]:.4f}")

print("\n" + "=" * 88)
print("有效通信距离 D(T)：算术 vs 几何（复刻团队保序+对数插值+首次穿阈）")
print("=" * 88)
hdr = f"{'call':<6}" + "".join(f"{T:>20}" for T in ["D(0.5)算术", "D(0.5)几何", "D(0.6)算术", "D(0.6)几何", "D(0.7)算术", "D(0.7)几何"])
print(hdr)
for ci, c in enumerate(CALLS):
    row = f"{c:<6}"
    for T in LEVELS:
        row += f"{crossing(DIST, ari[ci], T):>18.2f}{crossing(DIST, geo[ci], T):>18.2f}"
    print(row)

# ---- 5) 原始观测水平（不做保序）的 256m 排序 ----
print("\n" + "=" * 88)
print("256 m 原始观测均值下的排序（不拟合，直接看三通道合成）")
print("=" * 88)
obs_ari = obs.mean(axis=1)[:, -1]
obs_geo = np.exp(np.mean(np.log(obs), axis=1))[:, -1]
for ci, c in enumerate(CALLS):
    print(f"  {c:<5} 算术 {obs_ari[ci]:.4f}   几何 {obs_geo[ci]:.4f}")
print(f"  算术排序: {[CALLS[i] for i in np.argsort(-obs_ari)]}")
print(f"  几何排序: {[CALLS[i] for i in np.argsort(-obs_geo)]}")
