# -*- coding: utf-8 -*-
"""Kruskal-Wallis 检验：六项共同声学特征在 flat/mod/tv 三组间的整体差异。
6 项检验做 Holm 校正；效应量用秩 epsilon^2 = H/(N-1)。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

DATA = Path(r"G:\亚太数模\2026APMCM\题目与要求\date\Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
FEAT6 = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
LABELS = ["flat", "mod", "tv"]


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    adj[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adj


def main():
    df = pd.read_csv(DATA)
    df["CallType"] = df.CallType.str.lower()
    N = len(df)

    print("Kruskal-Wallis（六项共同特征，三组；6 项 Holm 校正，效应量 epsilon^2 = H/(N-1)）")
    print("=" * 84)
    print(f"{'feature':<9}{'H':>8}{'p_raw':>9}{'p_holm':>9}{'eps^2':>9}  {'flat中位':>9}{'mod中位':>9}{'tv中位':>9}")
    rows = []
    for f in FEAT6:
        groups = [df.loc[df.CallType.eq(l), f].dropna() for l in LABELS]
        H, p = kruskal(*groups)
        eps2 = H / (N - 1)
        meds = [g.median() for g in groups]
        rows.append((f, H, p, eps2, meds))
        print(f"{f:<9}{H:>8.2f}{p:>9.4f}{'—':>9}{eps2:>9.4f}  {meds[0]:>9.1f}{meds[1]:>9.1f}{meds[2]:>9.1f}")
    ph = holm([r[2] for r in rows])
    print("-" * 84)
    for (f, H, p, eps2, meds), phv in zip(rows, ph):
        mark = " ★" if phv < 0.05 else ""
        print(f"{f:<9}{H:>8.2f}{p:>9.4f}{phv:>9.4f}{eps2:>9.4f}  {'':>9}{'':>9}{'':>9}{mark}")

    out = Path(r"G:\亚太数模\非参数检验\kruskal_wallis.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature": f, "H": H, "p_raw": p, "p_holm": phv, "epsilon2": eps2,
                   "flat_median": m[0], "mod_median": m[1], "tv_median": m[2]}
                  for (f, H, p, eps2, m), phv in zip(rows, ph)]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[ok] -> {out}")


if __name__ == "__main__":
    main()
