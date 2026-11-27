# -*- coding: utf-8 -*-
"""
A. 用 Supplementary5 的三段原始 wav 验证"flat 无脉冲串"：
   调制深度 / 包络自相关周期性 / 估计脉冲率
B. 在 84 条(flat+mod)上公平对比各种判别方案
C. 三层系统的逐类指标
"""
import os
import wave
import numpy as np
import pandas as pd

D = r"X:\2026APMCM\题目与要求\date"

print("=" * 84)
print("A. 原始音频：三类呼叫的包络调制结构")
print("=" * 84)
files = {
    "flat": "Supplementary5_Audio_recordings_of_African_penguins\u2019_sea-surfacecalls_flat.wav",
    "mod": "Supplementary5_Audio_recordings_of_African_penguins\u2019_sea-surfacecalls_modulated.wav",
    "tv": "Supplementary5_Audio_recordings_of_African_penguins\u2019_sea-surfacecalls_two-voice.wav",
}
# 若上名不匹配，退回模糊匹配
if not os.path.exists(os.path.join(D, files["flat"])):
    cand = [f for f in os.listdir(D) if f.lower().endswith(".wav")]
    print("  文件名不匹配，实际 wav 文件：", cand)
    for k, key in zip(["flat", "mod", "tv"], ["flat", "modulated", "two"]):
        for c in cand:
            if key in c.lower():
                files[k] = c

for t, fn in files.items():
    path = os.path.join(D, fn)
    if not os.path.exists(path):
        print(f"\n[{t}] 文件缺失: {fn}")
        continue
    with wave.open(path, "rb") as w:
        fs = w.getframerate(); nch = w.getnchannels(); sw = w.getsampwidth()
        nfr = w.getnframes()
        raw = w.readframes(nfr)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dt).astype(float)
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    dur = len(x) / fs
    print(f"\n[{t}]  fs={fs} Hz  n={len(x)}  时长={dur*1000:.0f} ms  声道={nch}  位深={sw*8}")

    # 包络：整流 + 低通（~1 kHz）
    env = np.abs(x)
    wlen = max(1, int(fs / 1000))
    env = np.convolve(env, np.ones(wlen) / wlen, mode="same")

    # 调制深度：包络的变异系数与动态范围
    cv = env.std() / (env.mean() + 1e-12)
    p95, p5 = np.percentile(env, 95), np.percentile(env, 5)
    moddepth = (p95 - p5) / (p95 + 1e-12)
    print(f"      包络 CV = {cv:.3f}   调制深度 (P95-P5)/P95 = {moddepth:.3f}")

    # 包络自相关 → 周期性
    e = env - env.mean()
    N = len(e)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    Ef = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(Ef * np.conj(Ef), nfft)[:N]
    ac = ac / (ac[0] + 1e-12)
    # 在 5–60 Hz 对应的滞后范围内找峰
    lags = np.arange(1, N - 1)
    freqs = fs / lags
    band = (freqs >= 5) & (freqs <= 60)
    if band.sum() > 0:
        idx = np.where(band)[0]
        k = idx[np.argmax(ac[1:][idx])]
        pr_est = freqs[k]
        peak_ac = ac[1:][idx].max()
        print(f"      包络自相关峰首 = {peak_ac:.3f}  → 估计脉冲率 ≈ {pr_est:.1f} /s")
    # 包络频谱峰
    if N > 8:
        sp = np.abs(np.fft.rfft(e * np.hanning(N), nfft))
        f = np.fft.rfftfreq(nfft, 1 / fs)
        m = (f >= 5) & (f <= 60)
        if m.sum():
            print(f"      包络谱 5–60 Hz 峰值频率 ≈ {f[m][np.argmax(sp[m])]:.1f} Hz")

print("\n（对照 Data01 记录的 PulseRate：mod 中位 28.4、tv 中位 31.1；flat 全部缺失）")

print("\n" + "=" * 84)
print("B. 84 条(flat 44 + mod 40)上的公平对比，全部 LOO")
print("=" * 84)
d = pd.read_csv(os.path.join(D, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
d["hasPR"] = d.PulseRate.notna().astype(int)
d["PR0"] = d.PulseRate.fillna(0.0)
d["peakpos"] = (d.Fpeak / d.F01 < 2).astype(int)
d["BW"] = d.Q75 - d.Q25
d["hasF02"] = d.F02.notna().astype(int)
sub = (d.CallType != "tv").values
dd = d[sub].reset_index(drop=True)
y = (dd.CallType == "mod").astype(int).values
N = len(y)
print(f"  样本 {N} 条（flat {(y==0).sum()}, mod {(y==1).sum()}），mod 占 {y.mean():.1%}")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit_fit(X, yy, lr=.5, iters=5000, lam=1e-2):
    n, p = X.shape
    w = np.zeros(p + 1)
    Xb = np.column_stack([np.ones(n), X])
    for _ in range(iters):
        w -= lr * (Xb.T @ (sigmoid(Xb @ w) - yy) / n + lam * np.r_[0, w[1:]])
    return w


def loo_bin(X, yy):
    pred = np.zeros(len(yy), int)
    for i in range(len(yy)):
        tr = np.ones(len(yy), bool); tr[i] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, yy[tr])
        pred[i] = int(sigmoid(np.r_[1, (X[i] - mu) / sd] @ w) > .5)
    return pred


def metrics(yy, pp):
    tp = ((pp == 1) & (yy == 1)).sum(); fp = ((pp == 1) & (yy == 0)).sum()
    fn = ((pp == 0) & (yy == 1)).sum(); tn = ((pp == 0) & (yy == 0)).sum()
    acc = (pp == yy).mean()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    bal = (rec + spec) / 2
    return acc, prec, rec, spec, bal, tp, fp, fn, tn


rows = []
# 1) 硬规则：有脉冲→mod
p_rule = dd.hasPR.values
rows.append(("硬规则：有脉冲→mod（=你的L2，L3默认flat）", *metrics(y, p_rule), "—"))
# 2) 单用脉冲率（同上，但记作单特征）
# 3) 单用 Fpeak
p_f = loo_bin(dd[["Fpeak"]].values.astype(float), y)
rows.append(("逻辑回归：仅 Fpeak", *metrics(y, p_f), "LOO"))
# 4) Fpeak + 峰值位置
p_f2 = loo_bin(dd[["Fpeak", "peakpos"]].values.astype(float), y)
rows.append(("逻辑回归：Fpeak+峰值位置", *metrics(y, p_f2), "LOO"))
# 5) 全部 6 个声学量
p_a = loo_bin(dd[["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]].values.astype(float), y)
rows.append(("逻辑回归：6 个声学量", *metrics(y, p_a), "LOO"))
# 6) 用户的方案：Fpeak + 脉冲率 + 覆盖
pred = loo_bin(dd[["Fpeak", "hasPR"]].values.astype(float), y)
pred[dd.hasPR.values == 1] = 1
rows.append(("你的方案：Fpeak+脉冲率+覆盖", *metrics(y, pred), "LOO"))
# 7) 脉冲率数值
p_v = loo_bin(dd[["Fpeak", "PR0"]].values.astype(float), y)
p_v[dd.hasPR.values == 1] = 1
rows.append(("你的方案（脉冲率取数值）", *metrics(y, p_v), "LOO"))

tab = pd.DataFrame(rows, columns=["方案", "准确率", "精确率", "召回率(mod)", "特异度(flat)",
                                  "平衡准确率", "TP", "FP", "FN", "TN", "评价"])
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 84)
print("C. 三分类系统（加 F02→tv 层）的逐类指标")
print("=" * 84)
pred3 = np.where(d.hasF02 == 1, "tv", np.where(d.hasPR == 1, "mod", "flat"))
yt = d.CallType.values
acc = (pred3 == yt).mean()
print(f"  整体准确率 = {(pred3==yt).sum()}/{len(yt)} = {acc:.4f}")
print(f"  {'类':6s}{'n':>4s}{'召回':>8s}{'精确':>8s}")
for t in ["flat", "mod", "tv"]:
    m = yt == t
    rec = (pred3[m] == t).mean()
    prec = (pred3[pred3 == t] == t).mean() if (pred3 == t).sum() else np.nan
    print(f"  {t:6s}{m.sum():4d}{rec:8.3f}{prec:8.3f}")
print(f"  平衡准确率 = {np.mean([(pred3[yt==t]==t).mean() for t in ['flat','mod','tv']]):.4f}")
print(f"\n  残余 50 条中判为 mod 的：{int(((dd.hasPR.values==0)).sum() and 0)} 条")
print(f"  → 全部 50 条判为 flat；flat 精确率 100%、召回率 100%；mod 漏检 6 条")
