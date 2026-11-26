"""Shared statistics for Fig. 1 (three call types, Data01 clean data).

All numbers reported in the figure and in the analysis note are computed here
from the cleaned Data01 table; nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("X:/2026APMCM/APMCM_C_Preprocessing_Q1/Data01_clean.csv")

TYPES: Tuple[str, str, str] = ("flat", "mod", "tv")

# metric key -> (display label, unit)
METRICS: Dict[str, Tuple[str, str]] = {
    "Duration": ("Call duration", "s"),
    "F01": ("First fundamental frequency", "Hz"),
    "Fpeak": ("Peak frequency", "Hz"),
    "Q25": ("25% energy frequency", "Hz"),
    "Q50": ("50% energy frequency", "Hz"),
    "Q75": ("75% energy frequency", "Hz"),
}
COMMON = list(METRICS)
PAIRS = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["CallType"] = df["CallType"].astype(str).str.strip().str.lower()
    return df


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment (family-wise)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adj[idx] = min(1.0, running)
    return adj


def groups(df: pd.DataFrame, col: str, t: str) -> np.ndarray:
    return df.loc[df["CallType"] == t, col].dropna().to_numpy(dtype=float)


def kruskal_family(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in COMMON:
        arrs = [groups(df, key, t) for t in TYPES]
        h, p = stats.kruskal(*arrs)
        n_all = sum(a.size for a in arrs)
        rows.append(
            {
                "metric": key,
                "label": METRICS[key][0],
                "H": float(h),
                "df": len(arrs) - 1,
                "p_raw": float(p),
                "epsilon2_rank": float(h) / (n_all - 1),
                "n_total": int(n_all),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out


def pairwise_family(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in COMMON:
        for a, b in PAIRS:
            xa, xb = groups(df, key, a), groups(df, key, b)
            res = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            u = float(res.statistic)
            delta = 2.0 * u / (xa.size * xb.size) - 1.0
            rows.append(
                {
                    "metric": key,
                    "comparison": f"{a}/{b}",
                    "U": u,
                    "p_raw": float(res.pvalue),
                    "cliff_delta": delta,
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                    "median_a": float(np.median(xa)),
                    "median_b": float(np.median(xb)),
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out


def pulse_rate(df: pd.DataFrame) -> Dict[str, float]:
    mod = groups(df, "PulseRate", "mod")
    tv = groups(df, "PulseRate", "tv")
    res = stats.mannwhitneyu(mod, tv, alternative="two-sided")
    u = float(res.statistic)
    return {
        "n_mod": int(mod.size),
        "n_tv": int(tv.size),
        "median_mod": float(np.median(mod)),
        "median_tv": float(np.median(tv)),
        "U": u,
        "p_raw": float(res.pvalue),  # single pre-specified comparison, no Holm
        "cliff_delta": 2.0 * u / (mod.size * tv.size) - 1.0,
        "iqr_mod": float(np.subtract(*np.percentile(mod, [75, 25]))),
        "iqr_tv": float(np.subtract(*np.percentile(tv, [75, 25]))),
    }


def second_fundamental(df: pd.DataFrame) -> Dict[str, float]:
    tv = df.loc[df["CallType"] == "tv", ["F01", "F02"]].dropna()
    diff = tv["F02"].to_numpy() - tv["F01"].to_numpy()
    cleaner = int(np.sum(diff > 0))
    res = stats.wilcoxon(tv["F02"], tv["F01"], alternative="two-sided")
    return {
        "n": int(len(tv)),
        "n_f02_above_f01": cleaner,
        "median_diff": float(np.median(diff)),
        "wilcoxon_W": float(res.statistic),
        "wilcoxon_p": float(res.pvalue),
        "median_f01": float(tv["F01"].median()),
        "median_f02": float(tv["F02"].median()),
    }


def threshold_share(df: pd.DataFrame, col: str = "Fpeak", cut: float = 500.0) -> Dict[str, Tuple[int, int]]:
    out = {}
    for t in TYPES:
        v = groups(df, col, t)
        out[t] = (int(np.sum(v < cut)), int(v.size))
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in COMMON + ["F02", "PulseRate"]:
        for t in TYPES:
            v = groups(df, key, t)
            n_missing = int((df["CallType"] == t).sum() - v.size)
            if v.size == 0:
                rows.append({"metric": key, "call_type": t, "n": 0, "n_missing": n_missing})
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append(
                {
                    "metric": key,
                    "label": METRICS.get(key, (key, ""))[0],
                    "unit": METRICS.get(key, ("", ""))[1],
                    "call_type": t,
                    "n": int(v.size),
                    "n_missing": n_missing,
                    "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)),
                    "min": float(v.min()),
                    "p25": float(q1),
                    "median": float(med),
                    "p75": float(q3),
                    "iqr": float(q3 - q1),
                    "max": float(v.max()),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    kw = kruskal_family(df)
    pw = pairwise_family(df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(kw.round(4).to_string(index=False))
    print()
    print(pw.round(4).to_string(index=False))
    print()
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in pulse_rate(df).items()})
    print(second_fundamental(df))
    print(threshold_share(df))
