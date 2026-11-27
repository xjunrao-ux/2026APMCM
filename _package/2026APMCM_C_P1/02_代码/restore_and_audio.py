# -*- coding: utf-8 -*-
"""从 6ec2d13 只读提取 wav / R 代码 / 补充材料到 _restored/，并分析包络调制结构。"""
import os
import subprocess
import wave
import numpy as np

REPO = r"X:\2026APMCM"
COMMIT = "6ec2d13"
SUB = "题目与要求/date"
OUT = os.path.join(REPO, "_restored")
os.makedirs(OUT, exist_ok=True)


def git(*args):
    return subprocess.run(["git", "-c", "core.quotepath=false", *args],
                          cwd=REPO, capture_output=True)


r = git("ls-tree", "-r", "--name-only", COMMIT, SUB + "/")
names = r.stdout.decode("utf-8", "replace").splitlines()
print(f"提交 {COMMIT} 中 {SUB}/ 共 {len(names)} 个文件")

want = [n for n in names if n.lower().endswith((".wav", ".pdf")) or
        n.endswith(".R") or "Code1" in n]
for n in want:
    base = os.path.basename(n)
    dst = os.path.join(OUT, base)
    if os.path.exists(dst):
        continue
    d = git("show", f"{COMMIT}:{n}")
    if d.returncode == 0:
        with open(dst, "wb") as f:
            f.write(d.stdout)
        print(f"  提取 {base}  ({len(d.stdout):,} bytes)")
    else:
        print(f"  失败 {base}: {d.stderr.decode('utf-8', 'replace')[:120]}")

print("\n" + "=" * 82)
print("原始音频：三类呼叫的包络结构")
print("=" * 82)


def envelope_stats(path):
    with wave.open(path, "rb") as w:
        fs, nch, sw, nfr = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(nfr)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dt).astype(float)
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    if len(x) == 0:
        return None
    x = x - x.mean()
    dur = len(x) / fs
    # 整流 + 1 kHz 低通得到包络
    env = np.abs(x)
    wl = max(1, int(fs / 1000))
    env = np.convolve(env, np.ones(wl) / wl, mode="same")
    cv = env.std() / (env.mean() + 1e-12)
    p95, p5 = np.percentile(env, 95), np.percentile(env, 5)
    depth = (p95 - p5) / (p95 + 1e-12)
    # 包络自相关找 5–60 Hz 周期
    e = env - env.mean()
    N = len(e)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    Ef = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(Ef * np.conj(Ef), nfft)[:N]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(1, N - 1)
    fr = fs / lags
    band = (fr >= 5) & (fr <= 60)
    pr, peak = np.nan, np.nan
    if band.sum() > 3:
        idx = np.where(band)[0]
        k = idx[np.argmax(ac[1:][idx])]
        pr, peak = fr[k], ac[1:][k]
    # 包络谱峰
    sp = np.abs(np.fft.rfft(e * np.hanning(N), nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    m = (f >= 5) & (f <= 60)
    fpeak = f[m][np.argmax(sp[m])] if m.sum() else np.nan
    # 过零率（近似基频）
    zcr = np.mean(np.abs(np.diff(np.signbit(x).astype(int)))) * fs / 2
    return dict(fs=fs, dur=dur, cv=cv, depth=depth, pr=pr, ac=peak,
                epk=fpeak, zcr=zcr, n=len(x))


for t, key in [("flat", "flat"), ("mod", "modulated"), ("tv", "two")]:
    cand = [f for f in os.listdir(OUT) if f.lower().endswith(".wav") and key in f.lower()]
    if not cand:
        print(f"\n[{t}] 未找到 wav")
        continue
    s = envelope_stats(os.path.join(OUT, cand[0]))
    if s is None:
        print(f"\n[{t}] 读取失败")
        continue
    print(f"\n[{t}]  文件: {cand[0]}")
    print(f"    fs={s['fs']} Hz   n={s['n']}   时长={s['dur']*1000:.0f} ms")
    print(f"    包络 CV = {s['cv']:.3f}      调制深度 = {s['depth']:.3f}")
    print(f"    包络自相关峰 = {s['ac']:.3f}   估计脉冲率 ≈ {s['pr']:.1f} /s")
    print(f"    包络谱峰 ≈ {s['epk']:.1f} Hz     过零率≈{s['zcr']:.0f} Hz")
