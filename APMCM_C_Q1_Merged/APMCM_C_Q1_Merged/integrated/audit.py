"""Verify shared source values and reproduce reconciliation tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "2026APMCM_C_P1" / "01_数据"
CLEAN = ROOT / "APMCM_C_Q1_Full"
TABLES = Path(__file__).resolve().parent / "tables"


def save(frame, filename):
    frame.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")


def raw_csv(number):
    path, = RAW.glob(f"Data{number:02}_*.csv")
    return path, pd.read_csv(path)


def verify_inputs():
    records = []
    for number in (1, 2):
        path, source = raw_csv(number)
        clean = pd.read_csv(CLEAN / "data" / f"Data{number:02}_clean.csv")
        source["Date"] = pd.to_datetime(source.Date, format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
        pd.testing.assert_frame_equal(source, clean[source.columns], check_dtype=False,
                                      atol=1e-11, rtol=1e-12)
        records.append({"input": path.name, "rows": len(source), "columns": len(source.columns),
                        "check": "Same columns and values after date normalization"})
    long = pd.read_csv(CLEAN / "data/Data03-11_propagation_long.csv")
    for number in range(3, 12):
        path, source = raw_csv(number)
        wide = long[long.source_file.eq(path.name)].pivot(index="source_row", columns="distance_m",
                                                        values="correlation")
        expected_distances = [int(col.removesuffix("m")) for col in source]
        assert list(wide.columns) == expected_distances
        assert list(wide.index) == list(range(1, len(source) + 1))
        np.testing.assert_allclose(source.to_numpy(), wide.to_numpy(), atol=1e-11,
                                   rtol=1e-12, equal_nan=True)
        records.append({"input": path.name, "rows": len(source), "columns": len(source.columns),
                        "check": "Same values and missing cells after reshaping"})
    save(pd.DataFrame(records), "source_equivalence.csv")


def peak_and_stats():
    _, data = raw_csv(1)
    ratio = data.Fpeak / data.F01
    features = data.assign(Fpeak_lt_500=data.Fpeak.lt(500), Fpeak_F01_lt_2=ratio.lt(2),
                           ratio_rounded_3=ratio.round().eq(3), ratio=ratio)
    peak = features.groupby("CallType").agg(n=("Fpeak", "size"), Fpeak_lt_500=("Fpeak_lt_500", "sum"),
                                            Fpeak_F01_lt_2=("Fpeak_F01_lt_2", "sum"),
                                            ratio_rounded_3=("ratio_rounded_3", "sum"),
                                            median_ratio=("ratio", "median")).reset_index()
    save(peak, "peak_definition_counts.csv")
    tests = pd.read_csv(CLEAN / "03_nonparametric/tables/univariate_tests.csv")
    indicators = tests.test.eq("KWH")
    tests.loc[indicators, "p_bh6"] = false_discovery_control(tests.loc[indicators, "p"])
    save(tests[["feature", "test", "statistic", "p", "p_holm7", "p_bh6"]], "adjustment_comparison.csv")
    binary = data[data.CallType.isin(["flat", "mod"])]
    y = binary.CallType.eq("mod")
    auc = [{"feature": field, "higher_values_predict_mod": roc_auc_score(y, binary[field]),
            "lower_values_predict_mod": roc_auc_score(y, -binary[field])}
           for field in ("F01", "Fpeak")]
    save(pd.DataFrame(auc), "raw_feature_auc_direction.csv")


def playback_and_threshold():
    _, rows = raw_csv(2)
    outcomes = rows.pivot(index="expID", columns="Type", values=["Distance", "Npenguins", "ResponseYesNo"])
    close = (outcomes["Distance"]["call"] - outcomes["Distance"]["control"]).abs().le(10)
    summary = [{"pairs": len(outcomes), "unequal_distance": outcomes["Distance"].nunique(axis=1).gt(1).sum(),
                "unequal_group_size": outcomes["Npenguins"].nunique(axis=1).gt(1).sum(),
                "pairs_within_10m": close.sum(),
                "natural_responses_within_10m": outcomes.loc[close, ("ResponseYesNo", "call")].sum(),
                "control_responses_within_10m": outcomes.loc[close, ("ResponseYesNo", "control")].sum()}]
    save(pd.DataFrame(summary), "playback_pair_audit.csv")
    prop = pd.read_csv(CLEAN / "04_function_and_transmission/tables/propagation_by_measure_distance.csv")
    spectrogram = prop[prop.measure.eq("spectrogram")].pivot(index="distance_m", columns="call_type", values="mean")
    thresholds = []
    for kind, means in spectrogram.items():
        for target in (.5, .7):
            upper = np.flatnonzero(means.to_numpy() < target)[0]
            d1, d2 = means.index[upper - 1:upper + 1]
            fraction = (means.loc[d1] - target) / (means.loc[d1] - means.loc[d2])
            thresholds.append({"call_type": kind, "threshold": target, "lower_m": d1, "upper_m": d2,
                               "log_distance_interpolation_m": d1 * (d2 / d1)**fraction})
    save(pd.DataFrame(thresholds), "spectrogram_interpolation_check.csv")


def main():
    TABLES.mkdir(exist_ok=True)
    verify_inputs()
    peak_and_stats()
    playback_and_threshold()
    print("Verified Data01, Data02 and all nine propagation matrices. Reconciliation tables written.")


if __name__ == "__main__":
    main()
