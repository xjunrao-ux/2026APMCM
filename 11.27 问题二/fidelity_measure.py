# -*- coding: utf-8 -*-
"""
问题2 第一步：构造随传播距离变化的声保真度 (acoustic fidelity) 指标。

输入：Data03~Data11 共 9 个 CSV（三种鸣声类型 × 三种相关性测度）
  声谱图相关性   spectrogram       Data03(flat) Data04(mod) Data05(tv)
  幅度谱相关性   amplitude spectrum Data06(flat) Data07(mod) Data08(tv)
  幅度包络相关性 amplitude envelope Data09(flat) Data10(mod) Data11(tv)

每个 CSV：第一行是 7 个距离 4/8/16/32/64/128/256 m，之后每行是一个鸣声样本
相对 1m 参考信号的相关系数。

做法：
 1) 先按 (鸣声类型, 测度, 距离) 求均值与标准差；
 2) 用加权几何平均 / 加权算术平均把三种相关性合成一个保真度 F(d)；
 3) 若同一鸣声类型的三个文件样本数一致，再做样本级逐行合成，给出均值±SD；
 4) 输出表格与 CSV。
"""
import os
import numpy as np
import pandas as pd

DATE_DIR = r"G:\亚太数模\2026APMCM\题目与要求\date"
OUT_DIR  = r"G:\亚太数模\2026APMCM\problem2\results"
os.makedirs(OUT_DIR, exist_ok=True)

DISTANCES = [4, 8, 16, 32, 64, 128, 256]

# 文件映射： (鸣声类型, 测度) -> 文件名
FILES = {
    ("flat", "spectrogram"):       "Data03_correlation1m_flat_call_spectrogram.csv",
    ("mod",  "spectrogram"):       "Data04_correlation1m_modulated_call_spectrogram.csv",
    ("tv",   "spectrogram"):       "Data05_correlation1m_twovoice_call_spectrogram.csv",
    ("flat", "amp_spectrum"):      "Data06_correlation1m_flat_call_amplitude_spectrum.csv",
    ("mod",  "amp_spectrum"):      "Data07_correlation1m_modulated_call_amplitude_spectrum.csv",
    ("tv",   "amp_spectrum"):      "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv",
    ("flat", "envelope"):          "Data09_correlation1m_flat_call_envelope.csv",
    ("mod",  "envelope"):          "Data10_correlation1m_modulated_call_envelope.csv",
    ("tv",   "envelope"):          "Data11_correlation1m_twovoice_call_envelope.csv",
}

CALL_TYPES = ["flat", "mod", "tv"]
MEASURES  = ["spectrogram", "amp_spectrum", "envelope"]

# 权重：w = {spectrogram, amp_spectrum, envelope}
W = {"spectrogram": 0.5, "amp_spectrum": 0.3, "envelope": 0.2}
assert abs(sum(W.values()) - 1.0) < 1e-9, "权重之和必须为 1"


def load_all():
    """读入全部 9 个文件，返回 { (type, measure): DataFrame(样本 x 7距离) }。"""
    data = {}
    for (ct, meas), fname in FILES.items():
        path = os.path.join(DATE_DIR, fname)
        df = pd.read_csv(path)  # 第一行自动作为表头：4m,8m,...
        df.columns = [str(c).replace("m", "") for c in df.columns]
        df.columns = [int(float(c)) for c in df.columns]  # 转成 4,8,16...
        data[(ct, meas)] = df.astype(float)
    return data


def main():
    data = load_all()

    # ---------- 1) 逐测度的均值/标准差 ----------
    print("=" * 100)
    print("各文件样本数（行数）与缺失值情况")
    print("=" * 100)
    rows_info = []
    for (ct, meas), df in data.items():
        rows_info.append({"call_type": ct, "measure": meas,
                          "n_samples": len(df), "n_missing": int(df.isna().sum().sum())})
    info_df = pd.DataFrame(rows_info)
    print(info_df.to_string(index=False))
    print()

    # ---------- 2) 逐测度、逐距离的均值 ----------
    stat_rows = []
    for ct in CALL_TYPES:
        for d in DISTANCES:
            row = {"call_type": ct, "distance": d}
            for meas in MEASURES:
                col = data[(ct, meas)][d]
                row[f"r_{meas}"] = np.nanmean(col)
                row[f"sd_{meas}"] = np.nanstd(col)
                row[f"n_{meas}"] = int(col.notna().sum())
            stat_rows.append(row)
    stat = pd.DataFrame(stat_rows)

    # ---------- 3) 合成保真度 ----------
    # 几何平均（乘性，"且"语义：任一维度失真都拉低保真度）
    stat["F_geo"] = (
        stat["r_spectrogram"] ** W["spectrogram"]
        * stat["r_amp_spectrum"] ** W["amp_spectrum"]
        * stat["r_envelope"] ** W["envelope"]
    )
    # 算术平均（加性，补偿语义：各维度可相互补偿）
    stat["F_ari"] = (
        W["spectrogram"] * stat["r_spectrogram"]
        + W["amp_spectrum"] * stat["r_amp_spectrum"]
        + W["envelope"] * stat["r_envelope"]
    )

    # ---------- 4) 样本级逐行合成（仅当三文件样本数一致） ----------
    sample_rows = []
    for ct in CALL_TYPES:
        n = [len(data[(ct, m)]) for m in MEASURES]
        if len(set(n)) != 1:
            print(f"[提示] {ct} 三种测度的样本数不一致 {n}，跳过样本级逐行合成，只用均值级。")
            continue
        # 逐样本对齐后合成
        aligned = pd.concat(
            [data[(ct, m)].stack() for m in MEASURES], axis=1
        )
        aligned.columns = MEASURES
        aligned = aligned.dropna()
        geo = (
            aligned["spectrogram"] ** W["spectrogram"]
            * aligned["amp_spectrum"] ** W["amp_spectrum"]
            * aligned["envelope"] ** W["envelope"]
        )
        ari = (
            W["spectrogram"] * aligned["spectrogram"]
            + W["amp_spectrum"] * aligned["amp_spectrum"]
            + W["envelope"] * aligned["envelope"]
        )
        # aligned.index 是 (样本号, 距离) 的 MultiIndex
        idx = aligned.index.to_frame()
        idx.columns = ["sample", "distance"]
        tmp = pd.DataFrame({
            "call_type": ct,
            "distance": idx["distance"].astype(int),
            "F_geo_sample": geo.values,
            "F_ari_sample": ari.values,
        })
        g = tmp.groupby("distance").agg(
            F_geo_mean=("F_geo_sample", "mean"),
            F_geo_sd=("F_geo_sample", "std"),
            F_ari_mean=("F_ari_sample", "mean"),
            F_ari_sd=("F_ari_sample", "std"),
        ).reset_index()
        g["call_type"] = ct
        sample_rows.append(g)

    # 合并样本级结果到 stat（方便对照）
    if sample_rows:
        sample_stat = pd.concat(sample_rows, ignore_index=True)
        stat = stat.merge(sample_stat, on=["call_type", "distance"], how="left")

    # ---------- 输出 ----------
    cols = ["call_type", "distance",
            "r_spectrogram", "r_amp_spectrum", "r_envelope",
            "F_geo", "F_ari"]
    print("=" * 100)
    print(f"声保真度指标（权重 w = 声谱图{W['spectrogram']} / 幅度谱{W['amp_spectrum']} / 包络{W['envelope']}）")
    print("=" * 100)
    for ct in CALL_TYPES:
        sub = stat[stat["call_type"] == ct]
        print(f"\n【{ct}】")
        print(sub[cols].round(4).to_string(index=False))
        if "F_geo_mean" in stat.columns:
            sub2 = sub[["distance", "F_geo_mean", "F_geo_sd"]].copy()
            print(f"  (样本级几何平均保真度 均值±SD)：")
            print("  " + sub2.round(4).to_string(index=False).replace("\n", "\n  "))

    # 保存
    stat.to_csv(os.path.join(OUT_DIR, "fidelity_by_distance.csv"), index=False)
    print(f"\n结果已保存到 {os.path.join(OUT_DIR, 'fidelity_by_distance.csv')}")

    # ---------- 5) 权重敏感性（快速看几组权重下 F 的变化幅度） ----------
    print("\n" + "=" * 100)
    print("权重敏感性：改变三权重后，各鸣声类型在 4m 与 256m 的几何保真度")
    print("=" * 100)
    alt_weights = [
        {"name": "等权(1/3)",       "w": {"spectrogram": 1/3, "amp_spectrum": 1/3, "envelope": 1/3}},
        {"name": "基准(0.5/0.3/0.2)", "w": W},
        {"name": "强调频谱(0.4/0.4/0.2)", "w": {"spectrogram": 0.4, "amp_spectrum": 0.4, "envelope": 0.2}},
        {"name": "强调声谱图(0.7/0.2/0.1)", "w": {"spectrogram": 0.7, "amp_spectrum": 0.2, "envelope": 0.1}},
    ]
    sens_rows = []
    for alt in alt_weights:
        w = alt["w"]
        for ct in CALL_TYPES:
            sub = stat[stat["call_type"] == ct]
            for d in [4, 256]:
                r = sub[sub["distance"] == d].iloc[0]
                f = (r["r_spectrogram"] ** w["spectrogram"]
                     * r["r_amp_spectrum"] ** w["amp_spectrum"]
                     * r["r_envelope"] ** w["envelope"])
                sens_rows.append({"weight_scheme": alt["name"], "call_type": ct,
                                  "distance": d, "F_geo": f})
    sens = pd.DataFrame(sens_rows)
    piv = sens.pivot_table(index=["weight_scheme", "call_type"], columns="distance",
                           values="F_geo").round(4)
    print(piv.to_string())
    sens.to_csv(os.path.join(OUT_DIR, "fidelity_weight_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
