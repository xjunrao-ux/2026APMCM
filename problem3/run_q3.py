"""Run Problem 3 from any working directory with Python 3.11+, numpy, pandas, scipy."""
import os
for var in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"]:
    os.environ[var] = "1"
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

from models import BEHAVIORS, BITS, CALLS, COMBOS, CONTEXTS, combo_design, design, fit, predict
from validation import comparisons, cross_validate, folds, overall_checks, paired_tests, select

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv"
OUT = HERE / "results"
SEED = 20261127
TARGETS = ["HeadsUpDuration","VocalNcalls","ApproachDuration"]
KEYS = ["CallType","Distance","Context","Npenguins"]
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
    save(pd.concat(issues,ignore_index=True),"audit_discrepancies")
    d["code"]=d[[h+"YesNo" for h in BEHAVIORS]].to_numpy() @ BITS
    counts = pd.crosstab(d.Type,d.code).reindex(columns=range(8),fill_value=0)
    counts.columns=COMBOS
    save(counts.reset_index(),"combo_counts")
    blocks=d.groupby(["Type","CallType","Context"]).agg(n=("expID","size"),responses=("ResponseYesNo","sum"),
                     distance_min=("Distance","min"),distance_max=("Distance","max")).reset_index()
    save(blocks,"design_audit")
    tracks=d.groupby(["CallType","PlaybackTrack"]).size().rename("n").reset_index()
    save(tracks,"track_audit")
    base_match=(d.VocalRelNCalls==np.maximum(0,d.VocalNcalls-d.VocalNcallsBefore)).sum()
    mismatch=(d.ResponseYesNo.ne(d.code>0)).sum()
    if mismatch:
        save(raw.loc[d.ResponseYesNo.ne(d.code>0)],"overall_flag_discrepancies")
    text=f"""# Data02 数据审计

原始表 {len(d)} 行、{len(raw.columns)} 列；{len(pairs)} 个实验均为一次自然声和一次控制声。组内有 {pairs.Distance.nunique().gt(1).sum()} 对距离改变，{pairs.Npenguins.nunique().gt(1).sum()} 对企鹅数改变。距离范围 {d.Distance.min()}–{d.Distance.max()} m，中位数 {d.Distance.median():g} m；群体范围 {d.Npenguins.min()}–{d.Npenguins.max()} 只，中位数 {d.Npenguins.median():g} 只。全部 {d.PlaybackTrack.nunique()} 个音轨标识均有记录，跨类型重名 {tracks.groupby('PlaybackTrack').size().gt(1).sum()} 个；控制声仅一条，跨音轨推广受限。

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
    by=time.reshape(-1,3,3,6)
    p=by[:,:,:,0]*weights[None,:,None]
    event=p.sum(axis=1)
    mu,sig=by[:,:,:,4],by[:,:,:,5]
    denom=ndtr((np.log(60)-mu)/sig)
    low=np.zeros_like(event)
    high=low+60
    for _ in range(33):
        mid=(low+high)/2
        cdf=(p*ndtr((np.log(mid[:,None,:])-mu)/sig)/denom).sum(axis=1)
        smaller=cdf<.5*event
        low,high=np.where(smaller,mid,low),np.where(smaller,high,mid)
    rmst=(by[:,:,:,3]*weights[None,:,None]).sum(axis=1)
    return np.stack([event,1-event,(low+high)/2,rmst],axis=-1)


def predictions(d,grid,standard,ctx_weights,model,structure,family):
    overall,cp,q,tm,im=predict(grid,model,structure,family)
    b,gate,classes,_,_=model
    x,z=standard
    from scipy.special import softmax
    s=expit(z@b).reshape(4,-1).mean(axis=1)
    p=expit(z@gate)
    scp=np.column_stack([1-p,p[:,None]*softmax(combo_design(x)@classes,axis=1)]).reshape(4,len(d),8).mean(axis=1)
    standardized=comparisons(s,scp)
    marginal=[]
    for var,step in [("Distance",10),("Npenguins",1)]:
        inside=d[var]<=d[var].max()-step
        orig=d.loc[inside]
        modified=orig.assign(**{var:orig[var]+step})
        marginal.append((expit(design(modified,structure)@b)-expit(design(orig,structure)@b)).mean())
    return [overall[:,None],cp[:,:,None],q[:,:,None],tm[:,:,:4],im,integrate_context(tm,ctx_weights),
            standardized,np.column_stack([s,scp]),np.array(marginal)[:,None]]


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


def main(n_boot):
    OUT.mkdir(exist_ok=True)
    d=pd.read_csv(DATA)
    audit(d)
    splits=folds(d,SEED)
    scores,structure,family=select(d,splits)
    save(scores,"candidate_cv")
    val,times=cross_validate(d,splits,structure,family)
    save(val,"grouped_oof")
    save(times,"survival_cv")
    weights=np.ones(len(d))
    model=fit(d,weights,structure,family)
    grid=pd.DataFrame(product(CALLS,[50,100,150],CONTEXTS),columns=KEYS[:3])
    grid["Npenguins"]=d.Npenguins.median()
    grid["VocalNcallsBefore"]=d.VocalNcallsBefore.median()
    common=pd.concat([d.assign(CallType=c) for c in CALLS],ignore_index=True)
    standard=design(common),design(common,structure)
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
    # No timing estimate for an entirely unsupported treatment-behavior pair.
    for col in ["median_if_event_s","median_if_event_s_lo","median_if_event_s_hi"]:
        latency.loc[(latency.CallType=="control") & (latency.behavior=="Approach"),col]=np.nan
    save(latency,"q3_latency_predictions")
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
            if h=="Approach":
                ss.loc[ss.CallType=="control",h+"_median_if_event_s"]=np.nan
        sensitivity.append(ss)
    save(pd.concat(sensitivity),"sensitivity")
    score_p,glmm_sd,glmm=overall_checks(d,structure)
    plus,minus,paired_p,combo_p=paired_tests(d,SEED)
    from report import report
    report(d,avg,val,times,scores,cont,st,effects,structure,family,n_boot,plus,minus,
           paired_p,combo_p,score_p,glmm_sd,glmm,intensity)
    print("All prediction identities and grouped-fold checks passed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap",type=int,default=499)
    options=parser.parse_args()
    if options.bootstrap<2:
        parser.error("At least two resamples required")
    main(options.bootstrap)
