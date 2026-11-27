# -*- coding: utf-8 -*-
"""读取补充材料：Supplementary 4（传播退化图 PDF）文本 + Supplementary 5（三段音频）结构。"""
import pymupdf
import numpy as np
from scipy.io import wavfile
from scipy import signal
from pathlib import Path

DATE = Path(r"C:\Users\34232\Documents\GitHub\2026APMCM\题目与要求\date")

# ---- Supplementary 4 ----
s4 = DATE / "Supplementary4_Signal_degradation_estimation.pdf"
doc = pymupdf.open(str(s4))
print(f"== Supplementary4 PDF：{doc.page_count} 页 ==")
for i, page in enumerate(doc):
    txt = page.get_text().strip()
    print(f"\n--- 第 {i+1} 页 ---")
    print(txt[:1500] if txt else "(无文本，纯图)")

# ---- Supplementary 5 音频 ----
print("\n\n== Supplementary 5 三段音频结构 ==")
for name in ["flat", "modulated", "two-voice"]:
    files = list(DATE.glob(f"Supplementary5*{name}*.wav"))
    if not files:
        print(f"{name}: 未找到"); continue
    p = files[0]
    rate, data = wavfile.read(str(p))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(float)
    dur = len(data) / rate
    # 振幅包络（滑动 RMS）
    win = max(int(rate * 0.02), 1)
    rms = np.sqrt(np.convolve(data**2, np.ones(win)/win, mode="same"))
    # 频谱（平均幅度谱）
    freqs, _, Sxx = signal.spectrogram(data, fs=rate, nperseg=2048)
    mean_spec = Sxx.mean(axis=1)
    # 谱质心 / 峰频 / 分位数频率（幅度加权）
    f = freqs
    w = mean_spec / mean_spec.sum()
    centroid = float((f * w).sum())
    cum = np.cumsum(w)
    q25 = float(f[np.searchsorted(cum, 0.25)])
    q50 = float(f[np.searchsorted(cum, 0.50)])
    q75 = float(f[np.searchsorted(cum, 0.75)])
    fpeak = float(f[np.argmax(mean_spec)])
    # 包络起伏（脉冲率近似：包络自相关首个峰）
    env = rms - rms.mean()
    env /= (env.std() + 1e-9)
    ac = np.correlate(env, env, mode="full")[len(env)-1:]
    ac[:win] = 0
    peak_lag = int(np.argmax(ac[:rate]))
    pulse = rate / peak_lag if peak_lag > 0 else float("nan")
    print(f"\n{name}  ({p.name})")
    print(f"  采样率 {rate} Hz, 时长 {dur:.2f} s")
    print(f"  峰频 {fpeak:.0f} Hz, Q25/Q50/Q75 = {q25:.0f}/{q50:.0f}/{q75:.0f} Hz, 谱质心 {centroid:.0f} Hz")
    print(f"  包络主周期约 {peak_lag/rate*1000:.0f} ms → 近似脉冲率 {pulse:.1f} Hz")
