"""A. Read and audit the original call measurements."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "题目与要求" / "date"
OUT = ROOT / "results"
FIG = OUT / "figures"
LABELS = ["flat", "mod", "tv"]
COMMON = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
MEASURES = COMMON + ["F02", "PulseRate"]
UNITS = dict.fromkeys(MEASURES, "Hz")
UNITS.update(Duration="s", PulseRate="pulses/s")
COLORS = ["#267077", "#b25524", "#6852a0"]
SEED = 2026


def source(number):
    matches = list(DATA.glob(f"Data{number:02}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one Data{number:02} in {DATA}, found {matches}")
    return matches[0]


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return frame


def table(frame, digits=3):
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}g}"
        return str(value)
    lines = ["| " + " | ".join(frame.columns.map(str)) + " |",
             "|" + "---|" * len(frame.columns)]
    lines.extend("| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def report(name, body):
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")


def audit():
    FIG.mkdir(parents=True, exist_ok=True)
    path = source(1)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    raw_columns = list(df)
    for col in df:
        df[col] = df[col].str.strip().replace({"NA": np.nan, "": np.nan})
    df["CallType"] = df.CallType.str.lower()
    if set(df.CallType) != set(LABELS):
        raise ValueError("Unexpected or missing CallType.")
    df["Date"] = pd.to_datetime(df.Date, format="%d/%m/%Y", errors="raise").dt.strftime("%Y-%m-%d")
    for col in MEASURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[["Date", "Location"]].isna().any().any():
        raise ValueError("Missing grouping metadata.")
    dup = df.duplicated().sum()
    df.index = pd.RangeIndex(2, len(df) + 2, name="csv_line")
    counts = df.CallType.value_counts().reindex(LABELS)
    missing = df.groupby("CallType")[MEASURES].agg(lambda x: x.isna().sum())
    fields = pd.DataFrame({"field": raw_columns, "dtype": df.dtypes.astype(str).values,
                           "unit": [UNITS.get(c, "—") for c in raw_columns],
                           "missing": df.isna().sum().values, "missing_pct": 100 * df.isna().mean().values})
    dates = pd.crosstab(df.Date, df.CallType).reset_index()
    issues = []
    def mark(mask, field, rule):
        for idx in mask.index[mask.fillna(False)]:
            issues.append([idx, df.at[idx, "CallType"], field, rule, df.at[idx, field]])
    for c in MEASURES:
        mark(df[c].notna() & ~np.isfinite(df[c]), c, "physical: nonfinite")
        mark(df[c] < 0 if c == "PulseRate" else df[c] <= 0, c, "physical: range")
    mark((df.Q25 > df.Q50) | (df.Q50 > df.Q75), "Q50", "physical: quartile order")
    physical = len(issues)
    for group, block in [("all", df), *df.groupby("CallType")]:
        for c in MEASURES:
            x = block[c].dropna()
            if x.empty:
                continue
            q1, q3 = x.quantile([0.25, 0.75])
            mark((x < q1 - 1.5*(q3-q1)) | (x > q3 + 1.5*(q3-q1)), c, group + ": IQR")
            dev = (x - x.median()).abs()
            if dev.median() > 0:
                mark(0.67449 * dev / dev.median() > 3.5, c, group + ": MAD")
    flags = save(pd.DataFrame(issues, columns=["csv_line", "CallType", "field", "rule", "value"]), "audit_flags.csv")
    save(df.reset_index(), "clean_calls.csv")
    save(fields, "field_audit.csv")
    save(dates, "date_counts.csv")
    save(missing.reset_index(), "missing_by_type.csv")
    save(df[MEASURES].describe().T.rename_axis("feature").reset_index(), "measurement_ranges.csv")
    counts_table = pd.DataFrame({"CallType": counts.index, "n": counts.values, "%": 100*counts.values/len(df)})
    report("data_quality_report.md", f"""# A · 原始数据核验

输入为 {path.name}，UTF-8 兼容编码，逗号分隔。SHA-256 为 `{hashlib.sha256(path.read_bytes()).hexdigest()}`。原表 {len(df)} 行、{len(raw_columns)} 列。保留 CSV 原始行号以追溯；无企鹅个体、录音轨道或独立样本 ID。日期按日/月/年解析，原表出现日大于 12 的日期。

{table(counts_table)}

{table(fields)}

频率变量中 F01、F02 为第一和第二基频，Fpeak 为峰值频率，Q25、Q50、Q75 为频谱能量分位频率。Duration 为时长，PulseRate 为脉冲率。表头及题面字段表未标明单位，图表暂按秒、Hz、脉冲/秒解读，保留全部原始数值。

## 各类缺失数

{table(missing.reset_index())}

F02 的分布与双声结构相符。flat 的 PulseRate 整列无观测，mod 与 tv 中也有空值，CSV 无逐条缺失原因。保留 NA，不从空值推断脉冲率为零。

## 重复、边界与录音日期

标准化格式后全行重复 {dup} 条。按时长、频率正值，脉冲率非负，数值有限及频谱分位数有序检查，物理约束触发 {physical} 次。全体和类内分别应用 1.5 IQR 与稳健 Z 值绝对值大于 3.5 的 MAD 规则；共有 {flags.csv_line.nunique()} 行被至少一条规则标记。逐项见 audit_flags.csv。缺少录音对应的测量错误依据，保留原始极端观测。

共 {df.Date.nunique()} 个日期，地点记录数为 {df.Location.value_counts().to_dict()}。同日反复采样，按日期分组评估。地点样本不均衡，跨地点和跨个体泛化均待独立数据验证。

{table(dates)}
""")
    if physical or df[COMMON].isna().any().any():
        raise ValueError("Quality review required. See data_quality_report.md.")
    return df


if __name__ == "__main__":
    audit()
