"""Read the saved CSVs exactly as a downstream model would and check against Data02."""
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    output = here / "results"
    data = pd.read_csv(here.parent / "题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv")
    overall, combos, times, intensity = [
        pd.read_csv(output / f"q3_{name}_predictions.csv")
        for name in ["overall_response", "behavior_combo", "latency", "intensity"]
    ]
    assert [len(frame) for frame in [overall, combos, times, intensity]] == [36, 288, 144, 108]
    keys = ["CallType", "Distance", "Context", "Npenguins"]
    assert set(combos.combo) == {"000", "H", "V", "A", "HV", "HA", "VA", "HVA"}
    assert not combos.duplicated(keys + ["combo"]).any()
    assert not times.duplicated(keys + ["behavior"]).any()
    positive = combos[combos.combo != "000"].groupby(keys).p_all.sum().sort_index()
    assert np.allclose(positive, overall.set_index(keys).p_response.sort_index())
    assert times.Npenguins.eq(data.Npenguins.median()).all()
    for row in times.itertuples():
        exact = data[data.CallType.eq(row.CallType) & data.Distance.eq(row.Distance)
                     & data.Npenguins.eq(row.Npenguins)]
        if row.Context != "standardized":
            exact = exact[exact.Context.eq(row.Context)]
            cell = combos[combos.CallType.eq(row.CallType) & combos.Context.eq(row.Context)
                          & combos.Distance.eq(row.Distance)]
            letter = {"HeadsUp": "H", "Vocal": "V", "Approach": "A"}[row.behavior]
            assert np.isclose(row.p_event_60, cell.loc[cell.combo.str.contains(letter), "p_all"].sum())
        assert len(exact) == row.n_exact
        assert exact[row.behavior + "YesNo"].sum() == row.events_exact
        assert 60 * row.p_no_event_60 - 1e-7 <= row.rmst_60_s <= 60
    unsupported = times.CallType.eq("control") & times.behavior.eq("Approach")
    assert times.loc[unsupported, "median_if_event_s"].isna().all()
    partial_extrapolation = times.CallType.eq("mod") & times.Context.eq("standardized") & times.Distance.eq(150)
    assert times.loc[partial_extrapolation, "outside_distance_range"].all()
    for frame in [overall, times, combos, intensity]:
        for low in frame.filter(regex="_lo$"):
            high = low[:-3] + "_hi"
            assert frame[low].isna().equals(frame[high].isna())
            assert (frame[low].dropna() <= frame[high].dropna()).all()
    print("Verified four CSV interfaces against the original data and probability/time identities.")


if __name__ == "__main__":
    main()
