"""Run the Code1 baseline, paired playback models, and supplementary comparisons."""
import os
for var in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"]:
    os.environ[var] = "1"
import argparse
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from models import (BEHAVIORS, BITS, CALLS, COMBOS, CONTEXTS, WINDOW, combo_design,
                    design, fit, predict, window_time_cdf)
from validation import comparisons, cross_validate, folds, overall_checks, paired_tests, select
from supplements import acoustic_comparison, analysis_data, model_comparisons, run_code1

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv"
OUT = HERE / "results"
SEED = 20261127
TARGETS = ["HeadsUpDuration","VocalNcalls","ApproachDuration"]
KEYS = ["CallType","Distance","Context","Npenguins"]
HORIZONS = np.array([0, .5, 1, 1.5, 1.75, 2, 5, 10, 15, 30, 45, WINDOW])
PROTOCOL = "https://ore.eu/article/view/1538/version/2510"


def save(frame,name):
    frame.to_csv(OUT / (name+".csv"),index=False,encoding="utf-8-sig",float_format="%.10g")


def write(name,s):
    (OUT / (name+".md")).write_text(s.strip()+"\n",encoding="utf-8")


def audit(d):
    raw = pd.read_csv(DATA,keep_default_na=False,dtype=str)
    fields = []
    for col in raw:
        s = raw[col]
        vals = d[col].dropna()
        fields.append([col,str(d[col].dtype),(s=="").sum(),(s=="NA").sum(),d[col].isna().sum(),
                       (s=="0").sum(),vals.nunique(),vals.min() if len(vals) else None,
                       vals.max() if len(vals) else None])
    save(pd.DataFrame(fields,columns=["field","dtype","empty","literal_NA","missing","literal_zero",
                                       "distinct","min","max"]),"field_audit")
    pairs=d.groupby("expID")
    assert (pairs.size()==2).all() and (pairs.Type.nunique()==2).all()
    assert set(d.CallType)==set(CALLS) and set(d.Context)==set(CONTEXTS)
    assert d[["CallType","Context","Type","expID"]].notna().all().all()
    assert set(d.Type)=={"call","control"}
    assert (d.Type.eq("call")==d.CallType.ne("control")).all()
    needed=["Distance","Npenguins","ResponseYesNo","VocalNcallsBefore",*TARGETS]+[h+"YesNo" for h in BEHAVIORS]
    assert not d[needed].isna().any().any()
    assert d[[h+"YesNo" for h in BEHAVIORS]+["ResponseYesNo"]].isin([0,1]).all().all()
    for h in BEHAVIORS:
        t=d[h+"Latency"]
        assert t[d[h+"YesNo"]==1].between(0,60).all()
    issues = []
    for h,col in zip(BEHAVIORS,TARGETS):
        flag = d[h+"YesNo"].astype(bool)
        for field,positive in [(h+"Latency",d[h+"Latency"].notna()),(col,d[col]>0)]:
            mask=flag!=positive
            if mask.any():
                issues.append(raw.loc[mask].assign(csv_line=raw.index[mask]+2,field=field,
                              flag_used_for_time=d.loc[mask,h+"YesNo"],
                              analysis_time_s=np.where(flag[mask],d.loc[mask,h+"Latency"],60),
                              decision="原值保留；潜伏期按YesNo；强度按实测值"))
    empty_issues=raw.iloc[:0].reindex(columns=[*raw,"csv_line","field","flag_used_for_time",
                                               "analysis_time_s","decision"])
    save(pd.concat([empty_issues,*issues],ignore_index=True),"audit_discrepancies")
    assert np.array_equal(d.code, d[[h+"YesNo" for h in BEHAVIORS]].to_numpy() @ BITS)
    counts = pd.crosstab(d.Type,d.code).reindex(columns=range(8),fill_value=0)
    counts.columns=COMBOS
    save(counts.reset_index(),"combo_counts")
    blocks=d.groupby(["Type","CallType","Context"]).agg(n=("expID","size"),responses=("ResponseYesNo","sum"),
                     distance_min=("Distance","min"),distance_max=("Distance","max")).reset_index()
    save(blocks,"design_audit")
    tracks=d.groupby(["CallType","PlaybackTrack"]).size().rename("n").reset_index()
    save(tracks,"track_audit")
    base_match=(d.VocalRelNCalls==np.maximum(0,d.VocalNcalls-d.VocalNcallsBefore)).sum()
    size_by_arm=d.groupby("Type").Npenguins.median()
    mismatch=(d.ResponseYesNo.ne(d.code>0)).sum()
    if mismatch or (HERE/"results/overall_flag_discrepancies.csv").exists():
        save(raw.loc[d.ResponseYesNo.ne(d.code>0)],"overall_flag_discrepancies")
    text=f"""# Data02 数据审计

原始表 {len(d)} 行、{len(raw.columns)} 列；{len(pairs)} 个实验均为一次自然声和一次控制声。组内有 {pairs.Distance.nunique().gt(1).sum()} 对距离改变，{pairs.Npenguins.nunique().gt(1).sum()} 对企鹅数改变。距离范围 {d.Distance.min()}–{d.Distance.max()} m，中位数 {d.Distance.median():g} m；群体范围 {d.Npenguins.min()}–{d.Npenguins.max()} 只，中位数 {d.Npenguins.median():g} 只。全部 {d.PlaybackTrack.nunique()} 个音轨标识均有记录，跨类型重名 {tracks.groupby('PlaybackTrack').size().gt(1).sum()} 个；控制声仅一条，跨音轨推广受限。

补充一给出了 {len(pairs)} 个被试群体的规模分布。为让每个实验只计一次，分别在自然声和控制声记录中取中位数，结果为 {size_by_arm['call']:g} 只和 {size_by_arm['control']:g} 只。主模型使用各次回放的实测值，代表条件使用全表中位数 {d.Npenguins.median():g} 只。[群体规模分布图](../../题目与要求/date/Supplementary1_Distribution_of_African_penguin_group_sizes.pdf)

总体标记与任一目标行为不一致共 {mismatch} 行。八类频数见 combo_counts.csv，完整原始冲突记录见 audit_discrepancies.csv（同一行可能涉及两个字段）。抬头有 {((d.HeadsUpLatency==0)&(d.HeadsUpYesNo==1)).sum()} 条即时事件。接近标记为零却有潜伏期的记录有 {((d.ApproachYesNo==0)&d.ApproachLatency.notna()).sum()} 条，其潜伏期主分析记为 60 秒右删失。鸣叫标记为零却有时间的记录有 {((d.VocalYesNo==0)&d.VocalLatency.notna()).sum()} 条。替代编码结果见 sensitivity.csv。

逐行检查发现，VocalRelNCalls 在 {base_match}/{len(d)} 行等于 max(播放后次数−播放前次数, 0)，VocalYesNo 与 VocalRelNCalls>0 一致 {d.VocalYesNo.eq(d.VocalRelNCalls.gt(0)).sum()} 行。因此原始鸣叫次数为正与鸣叫响应标记可能不同。强度使用实际播放后次数，并控制播放前次数。VR1–VR19 是同一次播放的连续回应叫声标签，其缺失见 field_audit.csv。原始 NA、空字符串和字符 0 各自计数，原文件未改写。

各条件的观测量、距离覆盖见 design_audit.csv；音轨重复次数见 track_audit.csv。60 秒窗由原研究方法及表 1 确认。[实验协议]({PROTOCOL})
"""
    write("q3_data_audit",text)


def with_intervals(meta,values,boot,names):
    frame=meta.reset_index(drop=True).copy()
    for j,name in enumerate(names):
        frame[name]=values[...,j].ravel()
        lo,hi=np.quantile(boot[...,j],[.025,.975],axis=0)
        frame[name+"_lo"],frame[name+"_hi"]=lo.ravel(),hi.ravel()
    return frame


def integrate_context(time,weights):
    # Median of an incidence-weighted mixture, rather than mean of medians.
    by=time.reshape(-1,len(CONTEXTS),len(BEHAVIORS),6)
    p=by[:,:,:,0]*weights[None,:,None]
    event=p.sum(axis=1)
    mu,sig=by[:,:,:,4],by[:,:,:,5]
    low=np.zeros_like(event)
    high=low+WINDOW
    for _ in range(33):
        mid=(low+high)/2
        cdf=(p*window_time_cdf(mid[:,None,:],mu,sig)).sum(axis=1)
        smaller=cdf<.5*event
        low,high=np.where(smaller,mid,low),np.where(smaller,high,mid)
    rmst=(by[:,:,:,3]*weights[None,:,None]).sum(axis=1)
    return np.stack([event,1-event,(low+high)/2,rmst],axis=-1)


def latency_curves(time, weights):
    p60 = time[:, :, 0, None]
    conditional = window_time_cdf(HORIZONS, time[:, :, 4, None], time[:, :, 5, None])
    all_trials = p60 * conditional
    # grid rows are ordered by (call, distance, context); average joint
    # probabilities before conditioning on an event in the window.
    def mix(values):
        by_context = values.reshape(-1, len(CONTEXTS), *values.shape[1:])
        return np.einsum("c,gcbt->gbt", weights, by_context)
    mixed = mix(all_trials)
    joint = np.concatenate([all_trials, mixed])
    given_event = np.concatenate([conditional, mixed / mix(p60)])
    return np.stack([joint, given_event], axis=-1)


def trial_responses(d, model, structure, family):
    out = d[["expID","Type",*KEYS,"ResponseYesNo"]].copy()
    out["observed_combo"] = np.take(COMBOS, d.code)
    observed = pd.DataFrame({h: d[h+"Latency"].where(d[h+"YesNo"].eq(1))
                             for h in BEHAVIORS})
    out["first_observed_s"] = observed.min(axis=1)
    ties = observed.eq(out.first_observed_s, axis="index")
    out["first_behaviors"] = ["+".join(ties.columns[row]) for row in ties.to_numpy()]
    prob, combos, _, _, _ = predict(d,model,structure,family)
    out["fitted_p_response"] = prob
    out["fitted_p_any_behavior"] = 1 - combos[:,0]
    out["fitted_p_observed_combo"] = combos[np.arange(len(d)), d.code]
    return out


def predictions(d,grid,standard,ctx_weights,model,structure,family):
    overall,cp,q,tm,im=predict(grid,model,structure,family)
    b,gate,classes,_,_=model
    x,z,context_design=standard
    s=expit(z@b).reshape(4,-1).mean(axis=1)
    p=expit(z@gate)
    scp=np.column_stack([1-p,p[:,None]*softmax(combo_design(x)@classes,axis=1)]).reshape(4,len(d),8).mean(axis=1)
    standardized=comparisons(s,scp)
    # Hold distance and group size at the same empirical distribution across
    # all contexts, and use a common control in each within-context contrast.
    by_context = expit(context_design @ b).mean(axis=-1)
    natural = by_context[:,1:]
    arms = np.column_stack([natural,natural.mean(axis=1)])
    control = np.broadcast_to(by_context[:,:1],arms.shape)
    context_effects = np.stack([arms,control,arms-control],axis=-1)
    marginal=[]
    for var,step in [("Distance",10),("Npenguins",1)]:
        inside=d[var]<=d[var].max()-step
        orig=d.loc[inside]
        modified=orig.assign(**{var:orig[var]+step})
        marginal.append((expit(design(modified,structure)@b)-expit(design(orig,structure)@b)).mean())
    return [overall[:,None],cp[:,:,None],q[:,:,None],tm[:,:,:4],im,integrate_context(tm,ctx_weights),
            standardized,np.column_stack([s,scp]),np.array(marginal)[:,None],
            latency_curves(tm,ctx_weights),context_effects]


def support(d,meta):
    out=meta.copy()
    records=[]
    for row in out.itertuples():
        cells=d[d.CallType.eq(row.CallType)]
        if row.Context!="standardized":
            cells=cells[cells.Context.eq(row.Context)]
        exact=cells[cells.Distance.eq(row.Distance)&cells.Npenguins.eq(row.Npenguins)]
        at_d=cells[cells.Distance.eq(row.Distance)]
        h=row.behavior
        status="无该条件观测" if exact.empty else ("无响应" if exact[h+"YesNo"].sum()==0 else "观察到响应")
        observed=exact.loc[exact[h+"YesNo"]==1,h+"Latency"]
        ranges=cells.groupby("Context").Distance.agg(["min","max"])
        outside=((ranges["min"]>row.Distance)|(ranges["max"]<row.Distance)).any()
        records.append([len(exact),exact[h+"YesNo"].sum(),status,observed.median(),len(at_d),at_d[h+"YesNo"].sum(),
                        cells[h+"YesNo"].sum(),outside])
    cols=["n_exact","events_exact","observed_status","observed_median_if_event_s","n_at_distance_any_size","events_at_distance_any_size",
          "events_in_call_context_all_distances","outside_distance_range"]
    return pd.concat([out.reset_index(drop=True),pd.DataFrame(records,columns=cols)],axis=1)


def run_analysis(n_boot, rscript, stan_profile, q2_results):
    run_code1(DATA,OUT,"prepare",rscript)
    d=analysis_data(DATA,OUT)
    audit(d)
    splits=folds(d,SEED)
    scores,structure,family=select(d,splits)
    save(scores,"candidate_cv")
    val,times=cross_validate(d,splits,SEED)
    save(val,"grouped_oof")
    save(times,"survival_cv")
    weights=np.ones(len(d))
    model=fit(d,weights,structure,family)
    save(trial_responses(d,model,structure,family),"q3_trial_responses")
    grid=pd.DataFrame(product(CALLS,[50,100,150],CONTEXTS),columns=KEYS[:3])
    grid["Npenguins"]=d.Npenguins.median()
    grid["VocalNcallsBefore"]=d.VocalNcallsBefore.median()
    common=pd.concat([d.assign(CallType=c) for c in CALLS],ignore_index=True)
    context_design=np.stack([design(common.assign(Context=c),structure)
                             .reshape(len(CALLS),len(d),-1) for c in CONTEXTS])
    standard=design(common),design(common,structure),context_design
    ctx_weights=d.Context.value_counts(normalize=True).reindex(CONTEXTS).to_numpy()
    args=(d,grid,standard,ctx_weights)
    point=predictions(*args,model,structure,family)
    print(f"Grouped CV selected {structure}, {family}. Starting {n_boot} paired bootstraps.",flush=True)
    ids,inv=np.unique(d.expID,return_inverse=True)
    rng=np.random.default_rng(SEED)
    samples=[]
    for i in range(n_boot):
        w=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[inv]
        m=fit(d,w,structure,family)
        samples.append(predictions(*args,m,structure,family))
        if (i+1)%100==0:
            print(f"Completed {i+1} bootstrap fits",flush=True)
    boot=[np.stack(x) for x in zip(*samples)]
    def interval(j,meta,names):
        return with_intervals(meta,point[j],boot[j],names)
    save(interval(0,grid[KEYS],["p_response"]),"q3_overall_response_predictions")
    label=lambda names,key: pd.DataFrame({key:names})
    cm=grid[KEYS].merge(label(COMBOS,"combo"),how="cross")
    ct=grid[KEYS].merge(label(COMBOS[1:],"combo"),how="cross")
    cf=interval(1,cm,["p_all"]).merge(interval(2,ct,["p_given_any"]),on=KEYS+["combo"],how="left")
    save(cf,"q3_behavior_combo_predictions")
    tnames=["p_event_60","p_no_event_60","median_if_event_s","rmst_60_s"]
    labels=label(BEHAVIORS,"behavior")
    lat=interval(3,support(d,grid[KEYS].merge(labels,how="cross")),tnames)
    avggrid=grid[KEYS].drop_duplicates(["CallType","Distance"]).assign(Context="standardized")
    avg=interval(5,support(d,avggrid.merge(labels,how="cross")),tnames)
    latency=pd.concat([lat,avg],ignore_index=True)
    event_counts=d.groupby("CallType")[[h+"YesNo" for h in BEHAVIORS]].sum()
    call_behavior_n={(c,h): int(event_counts.loc[c,h+"YesNo"]) for c in CALLS for h in BEHAVIORS}
    latency["events_in_call_all_contexts"]=[call_behavior_n[c,h] for c,h in
                                            zip(latency.CallType,latency.behavior)]
    unsupported=latency.events_in_call_all_contexts.eq(0)
    latency.loc[unsupported,["median_if_event_s","median_if_event_s_lo","median_if_event_s_hi"]]=np.nan
    save(latency,"q3_latency_predictions")
    curve_keys=latency[[*KEYS,"behavior","events_in_call_all_contexts","outside_distance_range"]]
    curve_grid=curve_keys.merge(label(HORIZONS,"time_s"),how="cross")
    curves=interval(9,curve_grid,["p_event_by_t","p_by_t_given_event"])
    conditional_cols=["p_by_t_given_event"+suffix for suffix in ["","_lo","_hi"]]
    curves.loc[curves.events_in_call_all_contexts.eq(0),conditional_cols]=np.nan
    save(curves,"q3_latency_curves")
    intensity=interval(4,grid.merge(label(TARGETS,"measure"),how="cross"),
                       ["p_positive","mean_if_positive","expected_all"])
    save(intensity,"q3_intensity_predictions")
    contrast_meta=pd.DataFrame([("flat","control"),("mod","control"),("tv","control"),("natural_equal","control"),
                                 ("flat","mod"),("flat","tv"),("mod","tv")],columns=["first","second"])
    cont=interval(6,contrast_meta,["risk_difference","marginal_OR"]+["difference_"+c for c in COMBOS])
    save(cont,"q3_contrasts")
    st=interval(7,label(CALLS,"CallType"),["p_response"]+["p_"+c for c in COMBOS])
    save(st,"q3_standardized_by_call")
    effects=interval(8,label(["Distance_plus_10m","Npenguins_plus_1"],"change"),["risk_difference"])
    save(effects,"q3_covariate_effects")
    context_rows=pd.DataFrame(product(CONTEXTS,[*CALLS[1:],"natural_equal"]),
                              columns=["Context","CallType"])
    context_effects=interval(10,context_rows,["p_call","p_control","risk_difference"])
    save(context_effects,"q3_context_contrasts")
    assert np.allclose(cf.groupby(KEYS).p_all.sum(),1)
    assert np.allclose(cf.groupby(KEYS).p_given_any.sum(),1)
    assert np.allclose(latency.p_no_event_60+latency.p_event_60,1)
    assert np.allclose(intensity.p_positive*intensity.mean_if_positive,intensity.expected_all)
    assert latency.rmst_60_s.between(0,60).all() and val.expID.groupby(val.expID).size().eq(2).all()
    assert val.groupby("expID").fold.nunique().eq(1).all()
    sensitivity=[]
    for name,pen,alt in [("main",2.,False),("combo_ridge_half",1.,False),("combo_ridge_double",4.,False),
                          ("latency_present_as_event",2.,True)]:
        m=model if name=="main" else fit(d,weights,structure,family,pen,alt)
        _,cp,_,t,_=predict(grid,m,structure,family)
        ss=grid[KEYS].copy()
        ss["variant"]=name
        for j,c in enumerate(COMBOS):
            ss["p_"+c]=cp[:,j]
        for j,h in enumerate(BEHAVIORS):
            ss[h+"_p60"]=t[:,j,0]
            ss[h+"_median_if_event_s"]=t[:,j,2]
            event_flag=d[h+"Latency"].notna() if alt else d[h+"YesNo"].eq(1)
            ss.loc[~ss.CallType.isin(d.loc[event_flag,"CallType"]),h+"_median_if_event_s"]=np.nan
        sensitivity.append(ss)
    save(pd.concat(sensitivity),"sensitivity")
    acoustic=acoustic_comparison(HERE.parent,latency,q2_results)
    save(acoustic,"q3_supplementary4")
    run_code1(DATA,OUT,"fit",rscript,stan_profile)
    parameters,baseline=model_comparisons(DATA,OUT,intensity,latency)
    save(parameters,"q3_supplementary3_parameters")
    save(baseline,"q3_code1_comparison")
    score_p,glmm_sd,glmm=overall_checks(d,structure)
    plus,minus,paired_p,combo_p=paired_tests(d,SEED)
    from report import report
    report(d,avg,val,times,scores,cont,st,effects,structure,family,n_boot,plus,minus,
           paired_p,combo_p,score_p,glmm_sd,glmm,intensity,context_effects,curves,OUT)
    print("All prediction identities and grouped-fold checks passed.",flush=True)


def main(n_boot, rscript=None, stan_profile="published", q2_results=None):
    # Keep the last verified exports available while a new Stan run is in progress.
    from verify_exports import main as verify
    global OUT
    destination=OUT
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".q3-run-",dir=destination.parent) as work:
        OUT=Path(work)
        try:
            run_analysis(n_boot,rscript,stan_profile,q2_results)
            verify(q2_results,OUT)
            for result in OUT.iterdir():
                result.replace(destination/result.name)
        finally:
            OUT=destination


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap",type=int,default=499)
    parser.add_argument("--rscript",help="Rscript executable, also configurable with RSCRIPT")
    parser.add_argument("--stan-profile",choices=["short","published"],default="published",
                        help="4 chains: 2000/1000 or the published 20000/5000 iterations/warmup")
    parser.add_argument("--q2-results",type=Path,help="Directory with the Code2 GAM and Supplementary 4 check CSVs")
    options=parser.parse_args()
    if options.bootstrap<2:
        parser.error("At least two resamples required")
    main(options.bootstrap,options.rscript,options.stan_profile,options.q2_results)
