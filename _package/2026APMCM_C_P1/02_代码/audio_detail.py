# -*- coding: utf-8 -*-
"""细看三段 wav：包络脉动结构 + 时频谱（判断 flat 到底有没有脉冲串 / 是否等频）"""
import os
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = r"X:\2026APMCM\_restored"
FIG = r"X:\2026APMCM\figures"
os.makedirs(FIG, exist_ok=True)


def load(path):
    with wave.open(path, "rb") as w:
        fs, nch, sw, nfr = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(nfr)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dt).astype(float)
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    x -= x.mean()
    return x / (np.abs(x).max() + 1e-12), fs


def peaks_info(env, fs):
    """数包络峰（脉动个数）与间隔规律性"""
    thr = np.percentile(env, 60)
    above = env > thr
    # 上升沿进入
    idx = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(idx) < 2:
        return len(idx), np.nan, np.nan
    iv = np.diff(idx) / fs
    return len(idx), (1 / np.median(iv) if len(iv) else np.nan), (iv.std() / iv.mean() if len(iv) else np.nan)


files = {"flat": "flat", "mod": "modulated", "tv": "two"}
rows = []
fig, axes = plt.subplots(3, 2, figsize=(13.5, 8.4))
for i, (t, key) in enumerate(files.items()):
    cand = [f for f in os.listdir(OUT) if f.lower().endswith(".wav") and key in f.lower()]
    if not cand:
        continue
    x, fs = load(os.path.join(OUT, cand[0]))
    dur = len(x) / fs
    tt = np.arange(len(x)) / fs

    # 包络
    wl = max(1, int(fs / 1000))
    env = np.convolve(np.abs(x), np.ones(wl) / wl, mode="same")
    env /= env.max() + 1e-12
    n_pk, pr, cv_iv = peaks_info(env, fs)

    # 时频谱
    nper = 256; hop = 64
    win = np.hanning(nper)
    frames = range(0, len(x) - nper, hop)
    S = np.array([np.abs(np.fft.rfft(x[k:k + nper] * win)) for k in frames]).T
    S = 20 * np.log10(S + 1e-6)
    f = np.fft.rfftfreq(nper, 1 / fs)
    tt2 = np.array([k for k in frames]) / fs

    ax = axes[i, 0]
    ax.plot(tt * 1000, env, lw=.9, color="#4C72B0")
    ax.set_ylabel("归一化包络"); ax.set_title(
        f"({chr(97+i)}) {t}：包络峰 {n_pk} 个，间隔CV={cv_iv:.2f}", fontsize=10)
    if i == 2:
        ax.set_xlabel("时间 (ms)")
    ax.set_xlim(0, dur * 1000); ax.grid(alpha=.25)

    ax2 = axes[i, 1]
    m = f <= 3000
    im = ax2.pcolormesh(tt2 * 1000, f[m], S[m], shading="auto", cmap="magma")
    ax2.set_ylabel("频率 (Hz)"); ax2.set_title(f"{t}：时频谱", fontsize=10)
    if i == 2:
        ax2.set_xlabel("时间 (ms)")

    rows.append((t, dur * 1000, n_pk, pr, cv_iv))

plt.tight_layout()
p = os.path.join(FIG, "fig5_三类呼叫包络与时频谱.png")
plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print("[保存]", p)

print(f"\n{'类型':6s}{'时长ms':>9s}{'包络峰数':>9s}{'峰间隔率':>10s}{'间隔CV':>8s}")
for t, d, n, pr, cv in rows:
    print(f"{t:6s}{d:9.0f}{n:9d}{pr:10.1f}{cv:8.2f}")
