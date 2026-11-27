# -*- coding: utf-8 -*-
"""第三问第二小问（八种行为组合）结果作图，v2。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["svg.hashsalt"] = "q3"

RES = r"X:\2026APMCM\_q3_analysis\code1_results"
OUT = r"X:\2026APMCM\figures"
RAW = r"X:\2026APMCM\题目与要求\date\Data02_AfricanPenguins_playbacks_data.csv"
os.makedirs(OUT, exist_ok=True)

C = pd.read_csv(os.path.join(RES, "Q3_2_Code1_All_Contrasts.csv"), encoding="utf-8-sig")
S = pd.read_csv(os.path.join(RES, "Q3_2_Code1_Model_Sensitivity.csv"), encoding="utf-8-sig")
K = pd.read_csv(os.path.join(RES, "Q3_2_Code1_Context_Counts.csv"), encoding="utf-8-sig")
D = pd.read_csv(RAW)
D["grp"] = np.where(D.Type == "control", "control", D.CallType)

ORDER = ["No response", "H", "V", "A", "H+V", "H+A", "V+A", "H+V+A"]
ZH = {"No response": "无响应", "H": "仅抬头", "V": "仅鸣叫", "A": "仅靠近",
      "H+V": "抬头＋鸣叫", "H+A": "抬头＋靠近", "V+A": "鸣叫＋靠近", "H+V+A": "三种皆有"}
TCOL = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}
NAT, CTL = "#2C6FBB", "#9E9E9E"

eq = C[C.comparison_family == "equal mix vs control"].set_index("combination").loc[ORDER]
per = C[C.comparison_family == "each natural vs control"]
btw = C[C.comparison_family == "between natural sounds"]

n_by = D.groupby("grp").size()
r_by = D[D.ResponseYesNo == 1].groupby("grp").size()


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h) * 100, min(1, c + h) * 100


# ==========================================================================
# 图 1  八种组合的调整概率：对照 → 自然声
# ==========================================================================
fig, ax = plt.subplots(figsize=(7.6, 5.2))
yy = np.arange(len(ORDER))[::-1]
for i, cb in zip(yy, ORDER):
    q0, q1 = eq.loc[cb, "reference_probability_pct"], eq.loc[cb, "stimulus_probability_pct"]
    ax.plot([q0, q1], [i, i], color="#B0B0B0", lw=1.7, zorder=1, solid_capstyle="round")
    ax.scatter(q0, i, s=64, facecolor="white", edgecolor=CTL, lw=1.7, zorder=3)
    ax.scatter(q1, i, s=72, color=NAT, edgecolor="white", lw=0.8, zorder=4)
    d = eq.loc[cb, "probability_difference_pp"]
    lo, hi = eq.loc[cb, "track_simultaneous_95pct_low_pp"], eq.loc[cb, "track_simultaneous_95pct_high_pp"]
    sig = bool(eq.loc[cb, "track_interval_excludes_zero"])
    ax.text(99, i, f"{d:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", ha="left",
            fontsize=8.6, clip_on=False,
            fontweight="bold" if sig else "normal",
            color="#B03A2E" if sig else "#666666")
ax.set_yticks(yy)
ax.set_yticklabels([ZH[c] for c in ORDER], fontsize=10)
ax.set_xlim(0, 96)
ax.set_ylim(-0.65, len(ORDER) - 0.35)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xlabel("调整后概率（%）", fontsize=10)
ax.text(1.0, 1.02, "概率差（百分点）\n[95% 同时区间]", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8.4, color="#444444")
ax.grid(axis="x", color="#E8E8E8", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=CTL, mew=1.7,
                          ms=8, label="对照声音"),
                   Line2D([], [], marker="o", ls="", color=NAT, ms=8,
                          label="自然鸣声（三类等权）")],
          loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, frameon=False, fontsize=9)
ax.set_title("自然声把「无响应」压低 50 个百分点，增量集中在含鸣叫的组合",
             fontsize=12.5, pad=10, loc="left")
fig.text(0.012, -0.035, "红字 = 按录音轨道整组重抽样 3,000 次的同时区间不跨零；黑字 = 区间跨零",
         fontsize=8, color="#777777")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q3_2_fig1_组合概率平移.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 图 2  三个比较族的森林图
# ==========================================================================
fig = plt.figure(figsize=(12.6, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 0.95], wspace=0.62)

axa = fig.add_subplot(gs[0, 0])
yy = np.arange(len(ORDER))[::-1]
for i, cb in zip(yy, ORDER):
    d = eq.loc[cb, "probability_difference_pp"]
    lo, hi = eq.loc[cb, "track_simultaneous_95pct_low_pp"], eq.loc[cb, "track_simultaneous_95pct_high_pp"]
    sig = bool(eq.loc[cb, "track_interval_excludes_zero"])
    col = "#B03A2E" if (sig and d > 0) else ("#1F4E79" if (sig and d < 0) else "#8C8C8C")
    axa.plot([lo, hi], [i, i], color=col, lw=1.9, solid_capstyle="round", alpha=0.9)
    axa.scatter(d, i, s=52, color=col if sig else "white", edgecolor=col, lw=1.4, zorder=3)
    axa.text(40.5, i, f"{d:+.1f}", va="center", ha="left", fontsize=7.8, color="#444444")
axa.axvline(0, color="#333333", lw=1.0)
axa.set_yticks(yy)
axa.set_yticklabels([ZH[c] for c in ORDER], fontsize=9)
axa.set_xlim(-72, 44)
axa.set_ylim(-0.6, len(ORDER) - 0.4)
axa.set_xlabel("自然 − 对照（百分点）", fontsize=9.5)
axa.set_title("a  三类等权总体（8 项）", fontsize=10.5, loc="left")
axa.grid(axis="x", color="#EFEFEF", lw=0.8)
axa.set_axisbelow(True)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)

# ---- b. 各类鸣声（12 行合并标签）
axb = fig.add_subplot(gs[0, 1])
KEY = ["V", "H+V", "H+V+A", "No response"]
rows, labels = [], []
for cb in KEY:
    for t in ["flat", "mod", "tv"]:
        rows.append(per[(per.stimulus == t) & (per.combination == cb)].iloc[0])
        labels.append(f"{ZH[cb]} · {t}")
yy = np.arange(len(rows))[::-1]
for i, r, lab in zip(yy, rows, labels):
    c = TCOL[lab.split("· ")[1]]
    d, lo, hi = r.probability_difference_pp, r.track_simultaneous_95pct_low_pp, r.track_simultaneous_95pct_high_pp
    sig = bool(r.track_interval_excludes_zero)
    axb.plot([lo, hi], [i, i], color=c, lw=1.7, solid_capstyle="round")
    axb.scatter(d, i, s=44, color=c if sig else "white", edgecolor=c, lw=1.3, zorder=3)
    axb.text(40.5, i, f"{d:+.1f}", va="center", ha="left", fontsize=7.6, color="#444444")
axb.axvline(0, color="#333333", lw=1.0)
axb.set_yticks(yy)
axb.set_yticklabels(labels, fontsize=8.2)
axb.set_xlim(-72, 46)
axb.set_ylim(-0.6, len(rows) - 0.4)
axb.set_xlabel("自然 − 对照（百分点）", fontsize=9.5)
axb.set_title("b  三类鸣声各自对照（12 项）", fontsize=10.5, loc="left")
axb.grid(axis="x", color="#EFEFEF", lw=0.8)
axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
for j in (2, 5, 8):
    axb.axhline(j + 0.5, color="#DDDDDD", lw=0.8, zorder=0)

# ---- c. 三类之间：按组合分组，三色错位
axc = fig.add_subplot(gs[0, 2])
pairs = [("flat", "mod", -0.24), ("flat", "tv", 0.0), ("mod", "tv", 0.24)]
PCOL = {"flat-mod": "#4C72B0", "flat-tv": "#DD8452", "mod-tv": "#55A868"}
yy = np.arange(len(ORDER))[::-1]
for (a, b, off) in pairs:
    key = f"{a}-{b}"
    for i, cb in zip(yy, ORDER):
        r = btw[(btw.stimulus == a) & (btw.reference == b) & (btw.combination == cb)].iloc[0]
        d, lo, hi = r.probability_difference_pp, r.track_simultaneous_95pct_low_pp, r.track_simultaneous_95pct_high_pp
        y0 = i + off
        axc.plot([lo, hi], [y0, y0], color=PCOL[key], lw=1.2, alpha=0.75, solid_capstyle="round")
        axc.scatter(d, y0, s=26, color="white", edgecolor=PCOL[key], lw=1.1, zorder=3)
axc.axvline(0, color="#333333", lw=1.0)
axc.set_yticks(yy)
axc.set_yticklabels([ZH[c] for c in ORDER], fontsize=8.2)
axc.set_xlim(-24, 24)
axc.set_ylim(-0.6, len(ORDER) - 0.4)
axc.set_xticks([-20, -10, 0, 10, 20])
axc.set_xlabel("鸣声类型之差（百分点）", fontsize=9.5)
axc.set_title("c  三类之间两两比较（24 项）", fontsize=10.5, loc="left")
axc.legend(handles=[Line2D([], [], marker="o", ls="-", color=PCOL[f"{a}-{b}"], ms=5.5,
                           label=f"{a} − {b}") for a, b, _ in pairs],
           loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=3, frameon=False,
           fontsize=7.6, handletextpad=0.3, columnspacing=0.8)
axc.text(0.5, -0.30, "24 项区间全部跨零，最大差仅 +3.6 pp", transform=axc.transAxes,
         ha="center", va="top", fontsize=7.8, color="#B03A2E")
axc.grid(axis="x", color="#EFEFEF", lw=0.8)
axc.set_axisbelow(True)
for s in ("top", "right"):
    axc.spines[s].set_visible(False)

fig.suptitle("自然声提高的是「含鸣叫」的响应组合，而不是所有组合；三类之间无法排序",
             fontsize=12.5, x=0.01, ha="left", y=1.035)
fig.text(0.01, 0.985, "实心点＝95% 同时区间不跨零；空心点＝跨零。区间均由按录音轨道整组重抽样 3,000 次得到。",
         fontsize=8, color="#777777")
fig.savefig(os.path.join(OUT, "q3_2_fig2_森林图三族.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 图 3  已响应播放中的组合构成（两阶段分解）
# ==========================================================================
N_NAT = int(n_by.sum() - n_by["control"])
R_NAT = int(r_by.sum() - r_by["control"])
groups = [("对照声音", "control", eq["reference_probability_pct"]),
          ("自然声（等权）", "natural_all", eq["stimulus_probability_pct"])]
for t in ["flat", "mod", "tv"]:
    sub = per[per.stimulus == t].set_index("combination")
    groups.append((f"自然声 {t}", t, sub["stimulus_probability_pct"]))

SEG = ["H", "V", "H+V", "A", "H+A", "V+A", "H+V+A"]
SC = {"H": "#A6A6A6", "V": "#2E75B6", "H+V": "#7FB3D5", "A": "#E0A800",
      "H+A": "#F2C94C", "V+A": "#7F3C8D", "H+V+A": "#1B4F72"}

D["code"] = D.HeadsUpYesNo.astype(int) * 4 + D.VocalYesNo.astype(int) * 2 + D.ApproachYesNo.astype(int)
CODE = {}
for cb in SEG:
    c = 0
    for j, b in enumerate(["H", "V", "A"]):
        if b in cb.split("+"):
            c += 2 ** (2 - j)
    CODE[cb] = c


def sub_of(key):
    if key == "control":
        return D[D.Type == "control"]
    if key == "natural_all":
        return D[D.Type != "control"]
    return D[D.CallType == key]


GDEF = [("对照声音", "control"), ("自然声（等权）", "natural_all"),
        ("自然声 flat", "flat"), ("自然声 mod", "mod"), ("自然声 tv", "tv")]
rows = []
for name, key in GDEF:
    s = sub_of(key)
    r = s[s.ResponseYesNo == 1]
    rows.append((name, {cb: (r.code == CODE[cb]).mean() * 100 for cb in SEG}, len(r), len(s)))

fig, ax = plt.subplots(figsize=(8.6, 4.0))
yy = np.arange(len(rows))[::-1]
for i, (name, shares, r, n) in zip(yy, rows):
    left = 0
    for cb in SEG:
        w = shares[cb]
        ax.barh(i, w, left=left, color=SC[cb], edgecolor="white", lw=0.9, height=0.62)
        if w >= 7:
            ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center",
                    fontsize=8.8, color="white" if cb in ("V", "V+A", "H+V+A") else "#333333")
        left += w
    ax.text(101.5, i, f"{r}/{n} 次", va="center", fontsize=8.0, color="#666666")
ax.set_yticks(yy)
ax.set_yticklabels([g[0] for g in rows], fontsize=9.5)
ax.set_xlim(0, 118)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("占「已响应播放」的百分比（%，原始频数）", fontsize=9.5)
ax.set_title("同样是「有反应」：对照以仅抬头为主，自然声以含鸣叫的组合为主",
             fontsize=11.5, loc="left", pad=10)
ax.legend(handles=[Patch(facecolor=SC[c], label=ZH[c]) for c in SEG],
          loc="upper center", bbox_to_anchor=(0.47, -0.19), ncol=7, frameon=False,
          fontsize=8.4, handlelength=1.2, columnspacing=1.0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q3_2_fig3_已响应构成.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 图 4  模型敏感性与不确定性来源
# ==========================================================================
fig = plt.figure(figsize=(12.4, 3.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.90, 0.92], wspace=0.46)

SPECS = ["main", "baseline", "full_interaction", "baseline_and_full_interaction"]
SPEC_ZH = {"main": "原模型", "baseline": "+ 播放前鸣叫数",
           "full_interaction": "+ 情境×类型交互", "baseline_and_full_interaction": "两者均加"}
MK = {"main": "o", "baseline": "s", "full_interaction": "^", "baseline_and_full_interaction": "D"}
COL = {"main": "#111111", "baseline": "#2C6FBB", "full_interaction": "#DD8452",
       "baseline_and_full_interaction": "#55A868"}
OFFS = {"main": -0.24, "baseline": -0.08, "full_interaction": 0.08, "baseline_and_full_interaction": 0.24}

axa = fig.add_subplot(gs[0, 0])
KEEP = ["V", "H+V", "H+V+A"]
yy = np.arange(len(KEEP))[::-1]
for i, cb in zip(yy, KEEP):
    vals = {sp: S[(S.model == sp) & (S.combination == cb)]["natural_equal_minus_control_pp"].iloc[0]
            for sp in SPECS}
    base = vals["main"]
    for sp in SPECS:
        axa.scatter(vals[sp] - base, i + OFFS[sp], s=46, marker=MK[sp], color=COL[sp],
                    zorder=3, edgecolor="white", lw=0.6)
    axa.text(0.30, i, f"原模型 {base:+.1f} pp", va="center", ha="left", fontsize=8.0,
             color="#333333")
axa.axvline(0, color="#333333", lw=1.0)
axa.set_yticks(yy)
axa.set_yticklabels([ZH[c] for c in KEEP], fontsize=9.5)
axa.set_xlim(-0.32, 0.52)
axa.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
axa.set_xlabel("相对原模型的变化（百分点）", fontsize=9.2)
axa.set_title("a  四种规格相对原模型的变化 ≤ 0.15 pp", fontsize=10.5, loc="left")
axa.grid(axis="x", color="#EFEFEF", lw=0.8)
axa.set_axisbelow(True)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)
axa.legend(handles=[Line2D([], [], marker=MK[sp], ls="", color=COL[sp], ms=7,
                           label=SPEC_ZH[sp]) for sp in SPECS],
           loc="upper center", bbox_to_anchor=(0.42, -0.20), ncol=2, frameon=False,
           fontsize=8.0, handletextpad=0.3, labelspacing=0.3, columnspacing=0.9)

axb = fig.add_subplot(gs[0, 1])
loss = [S[S.model == sp]["track_grouped_cv_log_loss"].iloc[0] for sp in SPECS]
axb.bar(range(4), loss, color=[COL[sp] for sp in SPECS], width=0.62)
for i, v in enumerate(loss):
    axb.text(i, v + 0.00035, f"{v:.4f}", ha="center", fontsize=7.8, color="#333333")
axb.set_xticks(range(4))
axb.set_xticklabels(["原模型", "+基线", "+交互", "+两者"], fontsize=8.2)
axb.set_ylim(1.0695, 1.0813)
axb.set_ylabel("按轨道分组五折 CV 对数损失", fontsize=8.8)
axb.set_title("b  交互项略改善样本外拟合", fontsize=10.5, loc="left")
axb.grid(axis="y", color="#EFEFEF", lw=0.8)
axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.annotate("−0.0068", xy=(2, loss[2] - 0.00015), xytext=(1.22, 1.0766),
             fontsize=8.2, color="#B03A2E",
             arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=0.8))

axc = fig.add_subplot(gs[0, 2])
wt = (eq.track_simultaneous_95pct_high_pp - eq.track_simultaneous_95pct_low_pp).values
we = (eq.expid_simultaneous_95pct_high_pp - eq.expid_simultaneous_95pct_low_pp).values
lim = [4, 22]
axc.plot(lim, lim, color="#999999", lw=0.9, ls="--")
axc.scatter(we, wt, s=42, color=NAT, edgecolor="white", lw=0.7, zorder=3)
axc.set_xlim(*lim)
axc.set_ylim(*lim)
axc.set_xlabel("按实验重抽样的区间宽度（pp）", fontsize=9)
axc.set_ylabel("按录音轨道重抽样的区间宽度（pp）", fontsize=8.8)
axc.set_title("c  两种聚类口径的区间宽度", fontsize=10.5, loc="left")
axc.text(0.05, 0.94, "虚线以下 = 轨道口径更窄\n平均窄 1.8 pp（8 项中 6 项）",
         transform=axc.transAxes, fontsize=7.8, color="#555555", va="top")
axc.grid(color="#F2F2F2", lw=0.8)
axc.set_axisbelow(True)
for s in ("top", "right"):
    axc.spines[s].set_visible(False)

fig.suptitle("结论对模型规格、基线协变量与聚类单位都不敏感，代价是情境异质性被总体效应平均掉",
             fontsize=12, x=0.01, ha="left", y=1.02)
fig.savefig(os.path.join(OUT, "q3_2_fig4_敏感性.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 图 5  情境分层
# ==========================================================================
CTX = ["commuting", "searching", "after"]
CTX_ZH = {"commuting": "赶路\ncommuting", "searching": "搜索食物\nsearching", "after": "进食后\nafter"}
fig = plt.figure(figsize=(12.6, 4.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.95, 1.05], wspace=0.42)

axa = fig.add_subplot(gs[0, 0])
w = 0.36
DELTA = {}
for stim, off in [("control", -w / 2 - 0.02), ("natural_observed", w / 2 + 0.02)]:
    xs, hs, lo_e, hi_e = [], [], [], []
    for i, cx in enumerate(CTX):
        row = K[(K.context == cx) & (K.stimulus == stim)]
        n = int(row["all_playbacks_n"].iloc[0])
        k = n - int(row.iloc[0]["count"])
        lo, hi = wilson(k, n)
        xs.append(i + off)
        hs.append(k / n * 100)
        lo_e.append(k / n * 100 - lo)
        hi_e.append(hi - k / n * 100)
        DELTA.setdefault(cx, {})[stim] = k / n * 100
    axa.bar(xs, hs, width=w, color=CTL if stim == "control" else NAT,
            edgecolor="white", lw=0.8, label="对照声音" if stim == "control" else "自然声")
    axa.errorbar(xs, hs, yerr=[lo_e, hi_e], fmt="none", ecolor="#333333", lw=1.0, capsize=3)
    for x, h, hh in zip(xs, hs, hi_e):
        axa.text(x, h + hh + 2.0, f"{h:.0f}", ha="center", fontsize=8.4)
for i, cx in enumerate(CTX):
    dd = DELTA[cx]["natural_observed"] - DELTA[cx]["control"]
    axa.text(i, 112, f"Δ +{dd:.0f} pp", ha="center", fontsize=9.2, color="#B03A2E",
             fontweight="bold")
axa.set_xticks(range(3))
axa.set_xticklabels([CTX_ZH[c] for c in CTX], fontsize=8.6)
axa.set_ylim(0, 124)
axa.set_yticks([0, 25, 50, 75, 100])
axa.set_ylabel("任一行为出现的播放占比（%）", fontsize=9.2)
axa.set_title("a  灰＝对照，蓝＝自然声", fontsize=10.5, loc="left")
axa.grid(axis="y", color="#EFEFEF", lw=0.8)
axa.set_axisbelow(True)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)

axb = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(3)
labs = []
for i, cx in enumerate(CTX):
    v = D[(D.grp != "control") & (D.Context == cx)]["Npenguins"].values
    axb.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=15, color="#7F8C8D",
                alpha=0.5, lw=0)
    axb.plot([i - 0.30, i + 0.30], [np.median(v)] * 2, color="#B03A2E", lw=1.8)
    axb.text(i + 0.02, np.median(v) + 1.9, f"中位 {np.median(v):.0f}", ha="center",
             fontsize=8.0, color="#B03A2E",
             bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.88))
    labs.append(CTX_ZH[cx])
axb.axhline(8, color="#999999", lw=0.9, ls=":")
axb.set_xticks(range(3))
axb.set_xticklabels(labs, fontsize=8.4)
axb.set_ylabel("观测企鹅数量（只）", fontsize=9.2)
axb.set_ylim(0, 44)
axb.set_title("b 群体规模（虚线＝全体中位 8）", fontsize=10.5, loc="left")
axb.grid(axis="y", color="#EFEFEF", lw=0.8)
axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

axc = fig.add_subplot(gs[0, 2])
SEG2 = ["H", "V", "H+V", "A", "H+A", "V+A", "H+V+A", "No response"]
SC2 = dict(SC)
SC2["No response"] = "#E6E6E6"
yy = np.arange(3)[::-1]
for i, cx in zip(yy, CTX):
    sub = K[(K.context == cx) & (K.stimulus == "natural_observed")].set_index("combination")
    left = 0
    for cb in SEG2:
        v = sub.loc[cb, "percent_of_all_playbacks"]
        axc.barh(i, v, left=left, color=SC2[cb], edgecolor="white", lw=0.8, height=0.6)
        if cb == "No response" and v > 12:
            axc.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=8.4, color="#555555")
        left += v
axc.set_yticks(yy)
axc.set_yticklabels(["赶路", "搜索食物", "进食后"], fontsize=9)
axc.set_xlim(0, 100)
axc.set_xticks([0, 25, 50, 75, 100])
axc.set_xlabel("占该情境全部自然声播放的百分比（%）", fontsize=9)
axc.set_title("c  搜索食物情境的响应更多样", fontsize=10.5, loc="left")
for s in ("top", "right"):
    axc.spines[s].set_visible(False)

fig.suptitle("情境把自然声的响应率从 54% 拉到 95%，且与群体规模反向纠缠",
             fontsize=12, x=0.01, ha="left", y=1.02)
fig.savefig(os.path.join(OUT, "q3_2_fig5_情境分层.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
print("OK")
for cx in CTX:
    print(cx, {k: round(v, 1) for k, v in DELTA[cx].items()})
