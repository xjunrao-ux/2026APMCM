"""Read the saved CSVs exactly as a downstream model would and check against Data02."""
from pathlib import Path

import numpy as np
import pandas as pd

from models import BEHAVIORS, CALLS, CONTEXTS, WINDOW
from supplements import acoustic_comparison, analysis_data, published_parameters


def verify_curves(curves, times, data, keys):
    group = keys + ["behavior"]
    assert not curves.duplicated(group + ["time_s"]).any()
    horizons = sorted(curves.time_s.unique())
    assert horizons[0] == 0 and horizons[-1] == WINDOW
    assert len(curves) == len(times) * len(horizons)
    joined = curves.merge(times[group+["p_event_60","median_if_event_s"]],
                          on=group, how="left", validate="many_to_one", indicator=True)
    assert joined._merge.eq("both").all()
    supported = joined.events_in_call_all_contexts.gt(0)
    assert joined.p_by_t_given_event.notna().equals(supported)
    assert joined.p_event_by_t.between(0,1).all()
    np.testing.assert_allclose(joined.loc[supported,"p_by_t_given_event"] * joined.loc[supported,"p_event_60"],
                               joined.loc[supported,"p_event_by_t"],atol=1e-9)
    assert joined.loc[joined.time_s.eq(0),"p_event_by_t"].eq(0).all()
    end = joined[joined.time_s.eq(WINDOW)]
    np.testing.assert_allclose(end.p_event_by_t,end.p_event_60)
    for col in ["p_event_by_t","p_by_t_given_event"]:
        diffs = joined.sort_values("time_s").groupby(group)[col].diff().dropna()
        assert (diffs >= -1e-9).all()
    before = supported & (joined.time_s < joined.median_if_event_s)
    after = supported & (joined.time_s > joined.median_if_event_s)
    assert joined.loc[before,"p_by_t_given_event"].le(.5+1e-8).all()
    assert joined.loc[after,"p_by_t_given_event"].ge(.5-1e-8).all()
    weights = data.Context.value_counts(normalize=True)
    detail = curves[curves.Context.ne("standardized")].copy()
    detail["weighted"] = detail.Context.map(weights) * detail.p_event_by_t
    mixed_keys = ["CallType","Distance","Npenguins","behavior","time_s"]
    expected = detail.groupby(mixed_keys).weighted.sum().sort_index()
    actual = curves.query("Context=='standardized'").set_index(mixed_keys).p_event_by_t.sort_index()
    pd.testing.assert_index_equal(expected.index,actual.index)
    np.testing.assert_allclose(expected,actual,atol=1e-9)


def verify_trials(trials, data):
    observed = data.merge(trials,on=["expID","Type"],validate="one_to_one",
                          suffixes=("_raw","_fit"))
    assert len(observed)==len(data)==len(trials)
    for r in observed.itertuples():
        event_times = {h:getattr(r,h+"Latency") for h in BEHAVIORS if getattr(r,h+"YesNo")}
        combo = "".join(x for x,h in zip("HVA",BEHAVIORS) if h in event_times) or "000"
        assert r.observed_combo==combo
        if event_times:
            first=min(event_times.values())
            assert first==r.first_observed_s
            assert r.first_behaviors.split("+")==[h for h,t in event_times.items() if t==first]
        else:
            assert pd.isna(r.first_observed_s) and pd.isna(r.first_behaviors)
            assert np.isclose(r.fitted_p_observed_combo,1-r.fitted_p_any_behavior)
    assert trials.filter(regex="^fitted_p_").apply(lambda s:s.between(0,1).all()).all()


def verify_supplements(data_path, output, intensity, times, q2_results=None):
    d=analysis_data(data_path,output)
    flags=d[[h+"YesNo" for h in BEHAVIORS]].to_numpy() @ [4,2,1]
    np.testing.assert_array_equal(d.code,flags)
    terms=pd.read_csv(output/"q3_supplementary3_parameters.csv")
    publication=published_parameters(next(data_path.parent.glob("Supplementary3*.pdf")))
    coef_keys=["behavior","term"]
    pd.testing.assert_frame_equal(terms[publication.columns].sort_values(coef_keys).reset_index(drop=True),
                                  publication.sort_values(coef_keys).reset_index(drop=True))
    assert terms.groupby("behavior").size().to_dict()=={"Approach":14,"HeadsUp":18,"Vocal":19}
    rparams=pd.read_csv(output/"r_baseline_parameters.csv").set_index(coef_keys)
    coef=terms.set_index(coef_keys)
    np.testing.assert_allclose(coef[["mean","lo95","hi95"]],rparams.loc[coef.index,["mean","lo95","hi95"]])
    assert np.allclose(terms.refit_minus_published,terms["mean"]-terms.published_mean,atol=1e-8)
    diag=pd.read_csv(output/"r_baseline_diagnostics.csv").set_index("behavior")
    assert set(diag.index)==set(BEHAVIORS) and diag.chains.eq(4).all()
    assert (diag.iterations>diag.warmup).all() and (diag[["bulk_ess_min","tail_ess_min"]]>0).all().all()
    compared=pd.read_csv(output/"q3_code1_comparison.csv")
    keys=["CallType","Distance","Context","Npenguins","behavior"]
    assert len(compared)==108 and not compared.duplicated(keys).any()
    probs=["r_probability","r_lo95","r_hi95","q3_probability","q3_lo95","q3_hi95"]
    assert compared[probs].apply(lambda col: col.between(0,1)).all().all()
    assert np.allclose(compared.q3_minus_r_pp,100*(compared.q3_probability-compared.r_probability),atol=1e-7)
    rgrid=pd.read_csv(output/"r_baseline_predictions.csv").set_index(keys)
    indexed=compared.set_index(keys)
    np.testing.assert_allclose(indexed[probs[:3]],rgrid.loc[indexed.index,probs[:3]])
    for h, measure in zip(BEHAVIORS,["HeadsUpDuration","VocalNcalls","ApproachYesNo"]):
        rows=compared[compared.behavior.eq(h)].set_index(keys[:-1])
        if h=="Approach":
            source=times[(times.behavior==h)&(times.Context!="standardized")].set_index(keys[:-1])
            target="p_event_60"
        else:
            source=intensity[intensity.measure==measure].set_index(keys[:-1])
            target="p_positive"
            assert rows.groupby("CallType").r_probability.nunique().eq(1).all()
        np.testing.assert_allclose(rows[probs[3:]],source.loc[rows.index,[target,target+"_lo",target+"_hi"]])
    joined=pd.read_csv(output/"q3_supplementary4.csv")
    fresh=acoustic_comparison(data_path.parents[2],times,q2_results)
    pd.testing.assert_frame_equal(joined,fresh,check_dtype=False,atol=1e-8,rtol=1e-8)


def main(q2_results=None,output=None):
    here = Path(__file__).resolve().parent
    output = here / "results" if output is None else Path(output)
    data_path=here.parent / "题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv"
    data = pd.read_csv(data_path)
    overall, combos, times, intensity = [
        pd.read_csv(output / f"q3_{name}_predictions.csv")
        for name in ["overall_response", "behavior_combo", "latency", "intensity"]
    ]
    curves=pd.read_csv(output/"q3_latency_curves.csv")
    trials=pd.read_csv(output/"q3_trial_responses.csv",dtype={"observed_combo":str})
    context=pd.read_csv(output/"q3_context_contrasts.csv")
    oof=pd.read_csv(output/"grouped_oof.csv")
    assert [len(frame) for frame in [overall, combos, times, intensity]] == [36, 288, 144, 108]
    keys = ["CallType", "Distance", "Context", "Npenguins"]
    assert set(combos.combo) == {"000", "H", "V", "A", "HV", "HA", "VA", "HVA"}
    assert not combos.duplicated(keys + ["combo"]).any()
    assert not times.duplicated(keys + ["behavior"]).any()
    assert np.allclose(combos.groupby(keys).p_all.sum(),1)
    assert np.allclose(combos.groupby(keys).p_given_any.sum(),1)
    assert np.allclose(intensity.p_positive*intensity.mean_if_positive,intensity.expected_all)
    empty = combos[combos.combo=="000"].set_index(keys).p_all.sort_index()
    positive = combos[combos.combo!="000"].copy()
    expected_any = 1-positive.set_index(keys).index.map(empty).to_numpy()
    assert np.allclose(positive.p_all,expected_any*positive.p_given_any)
    flags = data[[h+"YesNo" for h in BEHAVIORS]].any(axis=1)
    if flags.eq(data.ResponseYesNo).all():
        assert np.allclose(1-empty,overall.set_index(keys).p_response.sort_index())
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
        actual=exact.loc[exact[row.behavior+"YesNo"].eq(1),row.behavior+"Latency"]
        assert np.isclose(actual.median(),row.observed_median_if_event_s,equal_nan=True)
        status="无该条件观测" if exact.empty else ("无响应" if actual.empty else "观察到响应")
        assert row.observed_status==status
        cell=data[data.CallType.eq(row.CallType)]
        assert cell[row.behavior+"YesNo"].sum()==row.events_in_call_all_contexts
        if row.Context!="standardized":
            cell=cell[cell.Context.eq(row.Context)]
        ranges=cell.groupby("Context").Distance.agg(["min","max"])
        assert row.outside_distance_range==((ranges["min"]>row.Distance)|(ranges["max"]<row.Distance)).any()
        assert 60 * row.p_no_event_60 - 1e-7 <= row.rmst_60_s <= 60
    assert times.median_if_event_s.isna().equals(times.events_in_call_all_contexts.eq(0))
    assert np.allclose(times.p_event_60+times.p_no_event_60,1)
    verify_curves(curves,times,data,keys)
    verify_trials(trials,data)
    assert set(context.Context)==set(CONTEXTS)
    assert len(context)==len(CONTEXTS)*len(CALLS)
    assert not context.duplicated(["Context","CallType"]).any()
    for _,g in context.groupby("Context"):
        arms=g.set_index("CallType")
        assert set(arms.index)=={*CALLS[1:],"natural_equal"}
        assert np.isclose(arms.loc[CALLS[1:],"p_call"].mean(),arms.loc["natural_equal","p_call"])
        assert g.p_control.nunique()==1
        assert np.allclose(g.p_call-g.p_control,g.risk_difference,atol=1e-9)
    assert len(oof)==len(oof.merge(data[["expID","Type"]],validate="one_to_one"))==len(data)
    assert oof.groupby("expID").fold.nunique().eq(1).all() and oof.fold.nunique()==5
    assert oof.groupby("fold")[["overall_structure","positive_count_family"]].nunique().eq(1).all().all()
    for frame in [overall, times, combos, intensity, curves, context]:
        for low in frame.filter(regex="_lo$"):
            high = low[:-3] + "_hi"
            assert frame[low].isna().equals(frame[high].isna())
            assert (frame[low].dropna() <= frame[high].dropna()).all()
    verify_supplements(data_path,output,intensity,times,q2_results)
    print(f"Verified {len(trials)} trials, {len(curves)} curve rows, {len(context)} context contrasts, "
          "grouped folds, 51 published coefficients, 108 Code1 comparisons, and 9 acoustic conditions.")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--q2-results",type=Path)
    main(parser.parse_args().q2_results)
