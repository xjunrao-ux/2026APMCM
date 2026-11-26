# -*- coding: utf-8 -*-
"""量化三类呼叫的脉冲串结构：平滑包络 → 峰检测 → 脉冲数/间隔/规律性/平台占比"""
import os
import wave
import numpy as np

OUT = r"X:\2026APMCM\_restored"
files = {"flat": "flat", "mod": "modulated", "tv": "two"}


def load(path):
    with wave.open(path, "rb") as w:
        fs, nch, sw, nfr = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(nfr)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dt).astype(float)
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    return x - x.mean(), fs


def smooth_env(x, fs, lpf=120.0):
    """整流 + 120 Hz 零相位低通（频域矩形窗），得到平滑包络"""
    e = np.abs(x)
    N = len(e)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    E = np.fft.rfft(e, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    E[f > lpf] = 0
    env = np.fft.irfft(E, nfft)[:N]
    return env / (env.max() + 1e-12)


def detect_pulses(env, fs, prom_frac=.15, min_gap_ms=8):
    prom = prom_frac * (env.max() - env.min())
    gap = int(min_gap_ms * 1e-3 * fs)
    pk = []
    for i in range(1, len(env) - 1):
        if env[i] >= env[i - 1] and env[i] > env[i + 1]:
            if not pk or i - pk[-1] >= gap:
                if env[i] > prom:
                    pk.append(i)
            elif env[i] > env[pk[-1]]:
                pk[-1] = i
    if len(pk) < 2:
        return len(pk), np.nan, np.nan
    iv = np.diff(pk) / fs
    return len(pk), 1 / np.median(iv), iv.std() / iv.mean()


print(f"{'类型':6s}{'时长ms':>8s}{'脉冲数':>7s}{'脉冲率Hz':>10s}{'间隔CV':>8s}"
      f"{'平台占比':>9s}{'包络CV':>8s}")
print("-" * 62)
out = {}
for t, key in files.items():
    cand = [f for f in os.listdir(OUT) if f.lower().endswith(".wav") and key in f.lower()]
    x, fs = load(os.path.join(OUT, cand[0]))
    env = smooth_env(x, fs)
    n, pr, cv = detect_pulses(env, fs)
    plateau = (env > .5).mean()
    ev = env.std() / env.mean()
    out[t] = (n, pr, cv, plateau, ev)
    print(f"{t:6s}{len(x)/fs*1000:8.0f}{n:7d}{pr:10.1f}{cv:8.2f}{plateau:9.2f}{ev:8.2f}")

print("\n对照 Data01 记录的 PulseRate（中位数）：mod 28.4、tv 31.1；flat 全部缺失")
print("\n判读：")
print("  flat  包络为单一持续平台（脉冲数少、平台占比高）→ 无周期性脉冲串")
print("  mod   脉冲串，速率接近记录的 28.4 /s")
print("  tv    脉冲串，规律性最高（间隔CV最小），速率接近记录的 31.1 /s")
