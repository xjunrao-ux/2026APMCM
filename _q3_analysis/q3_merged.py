"""Fit the consolidated C Q3 model and export its numerical results."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.special import expit, gammaln, log_ndtr, softmax
from scipy.stats import binomtest

from _fit_q3_work import (
    BEHAVIORS, CALLS, COMBO_SD, CON, CONTEXTS, CONTEXT_WEIGHT, CTX, CTYPE,
    D, DATA, INDICATORS, N, NAT, OCC_SD, PRE, Q, TRACK_ID, W, X, Y, YCODE,
    design, fit_count, logistic, optimize, serialize, simultaneous, softmax_fit,
)
from q3_behavior_factors import observed
from q3_latency_model import (
    ENDPOINTS, HELD_OUT, MIDPOINTS, TIME_SCALES, fit_endpoint, time_parameters,
    track_bootstrap, validate_time,
)
from q3_latency_scenarios import CONDITIONS, EVENTS, TIMES, observed_support, summarize


COMBOS = ("none", "A", "V", "VA", "H", "HA", "HV", "HVA")
STRUCTURES = ("log_additive", "log_pooled_context", "log_full_context",
              "linear_additive", "linear_full_context")
SCALES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
CONTRASTS = ("flat-control", "mod-control", "tv-control", "natural_equal-control",
             "mod-flat", "tv-flat", "tv-mod")
CONTEXT_MASK = (N >= 2) & (N <= 10) & (D >= 20) & (D <= 150)
STRENGTH_PRIOR = np.array([3, 2, 1, 1, .9, .9, .7, .7])


def features(call, context, distance, size, structure):
    base = design(call, context, distance, size)
    x = base[:, :8].copy()
    if structure.startswith("linear"):
        x[:, 6], x[:, 7] = (np.asarray(distance)-80)/50, (np.asarray(size)-8)/10
    if "pooled_context" in structure:
        return np.column_stack([x, base[:, 8:10]])
    if "full_context" in structure:
        interaction = (x[:, 1:4, None] * x[:, None, 4:6]).reshape(len(x), 6)
        return np.column_stack([x, interaction])
    return x


def fit_occurrence(weight, specifications, starts=(None, None)):
    gate_x, gate_sd = specifications[0]
    mix_x, mix_sd = specifications[1]
    use, hit = weight > 0, (weight > 0) & (YCODE > 0)
    return (logistic(gate_x[use], YCODE[use] > 0, weight[use], gate_sd, start=starts[0]),
            softmax_fit(mix_x[hit], YCODE[hit], weight[hit], mix_sd, start=starts[1]))


def select_occurrence():
    rows = []
    options = []
    event = YCODE > 0
    for name in STRUCTURES:
        x = features(CTYPE, CTX, D, N, name)
        base_sds = [np.r_[sd[:8], np.full(x.shape[1]-8, extra)]
                    for sd, extra in ((OCC_SD, .65), (COMBO_SD, .55))]
        for scale in SCALES:
            specs = [(x, sd*scale) for sd in base_sds]
            loss = np.zeros(2)
            forecast = np.zeros(len(x))
            for held in HELD_OUT:
                w = np.ones(len(x))
                w[held] = 0
                beta, gamma = fit_occurrence(w, specs)
                eta = x[held] @ beta
                forecast[held] = expit(eta)
                loss[0] += (np.logaddexp(0, eta)-event[held]*eta).sum()
                target = held[event[held]]
                cond = softmax(x[target] @ gamma, axis=1)
                loss[1] -= np.log(cond[np.arange(len(target)),YCODE[target]-1]).sum()
            rows.append({"structure": name, "scale": scale, "gate_nll": loss[0]/len(x),
                         "mix_nll_given_event": loss[1]/event.sum(),
                         "gate_brier": np.mean((forecast-event)**2)})
            options.append(specs)
    idx = [min(range(len(rows)), key=lambda i: rows[i][key])
           for key in ("gate_nll","mix_nll_given_event")]
    return [rows[i] for i in idx], [options[i][j] for j,i in enumerate(idx)], rows


def mix_predict(x, coefs):
    p = expit(x[0] @ coefs[0])
    conditional = softmax(x[1] @ coefs[1], axis=-1)
    return np.concatenate([(1-p)[...,None], p[...,None]*conditional], axis=-1)


def marginals(combo):
    return np.concatenate([(1-combo[..., :1]), combo @ INDICATORS], axis=-1)


def compare(v):
    return np.stack([v[1]-v[0], v[2]-v[0], v[3]-v[0], v[1:].mean(axis=0)-v[0],
                     v[2]-v[1], v[3]-v[1], v[3]-v[2]])


def prepare_grids(selection, size):
    structure = [row["structure"] for row in selection]

    def blocks(context, distance, group_size):
        return [np.stack([features(c,context,distance,group_size,s) for c in range(4)])
                for s in structure]

    observed_grid = blocks(CTX,D,N)
    factor_grids = {
        "contexts": [blocks(c,D[CONTEXT_MASK],N[CONTEXT_MASK]) for c in range(3)],
        "distance": [blocks(CTX,d,N) for d in (50,100)],
        "size": [blocks(CTX,D,n) for n in (4,8)],
    }
    calls = np.repeat([c for c,_ in CONDITIONS],3)
    distances = np.repeat([d for _,d in CONDITIONS],3)
    contexts = np.tile(np.arange(3),len(CONDITIONS))
    scenario = [features(calls,contexts,distances,size,s).reshape(12,3,-1) for s in structure]
    time_x = design(calls,contexts,distances,size).reshape(12,3,-1)
    return observed_grid, factor_grids, scenario, time_x


def time_selection():
    selected = []
    for i,(name,midpoint) in enumerate(zip(ENDPOINTS,MIDPOINTS)):
        terms = np.array([0,2,3,4,5,6,7]) if name=="Approach" else np.arange(8)
        scale, cv = validate_time(terms,TIMES[:,i],midpoint,(0.0,*TIME_SCALES))
        selected.append({"terms":terms if scale else terms[:1],"scale":scale,"cv":cv,
                         "events":np.isfinite(TIMES[:,i]).sum()})
    return selected


def truncated_poisson(x, y, w, start):
    def objective(b):
        eta = x@b
        mu = np.exp(eta)
        positive = -np.expm1(-mu)
        nll = mu-y*eta+gammaln(y+1)+np.log(positive)
        return w@nll, x.T@(w*(mu/positive-y))
    p = x.shape[1]
    return optimize(objective,np.zeros(p),np.r_[1.7,np.full(p-1,.5)],
                    start=start,bounds=[(-7,7)]*p)


def count_moments(x, coef, family):
    if family=="nb":
        mu, size = np.exp(x@coef[:-1]), np.exp(coef[-1])
        denominator = -np.expm1(-size*np.log1p(mu/size))
    else:
        mu = np.exp(x@coef)
        denominator = -np.expm1(-mu)
    return mu/denominator


def count_loss(x,y,coef,family):
    if family=="poisson":
        eta = x@coef
        mu = np.exp(eta)
        return mu-y*eta+gammaln(y+1)+np.log(-np.expm1(-mu))
    eta = x@coef[:-1]
    mu, size = np.exp(eta), np.exp(coef[-1])
    log0 = -size*np.log1p(mu/size)
    return (-gammaln(y+size)+gammaln(size)+gammaln(y+1)-log0
            -y*(eta-np.log(mu+size))+np.log(-np.expm1(log0)))


VOICE_X = np.column_stack([X[:,:8],np.log1p(PRE)])


def fit_positive_count(x,y,w,family,start=None):
    return (fit_count if family=="nb" else truncated_poisson)(x,y,w,start)


def select_count():
    y = Q[:,1]
    loss = {name:0.0 for name in ("poisson","nb")}
    for held in HELD_OUT:
        test = np.zeros(len(y),dtype=bool)
        test[held] = True
        train, valid = (~test)&(y>0), test&(y>0)
        for family in loss:
            fit = fit_positive_count(VOICE_X[train],y[train],np.ones(train.sum()),family)
            loss[family] += count_loss(VOICE_X[valid],y[valid],fit,family).sum()/(y>0).sum()
    return min(loss,key=loss.get),loss


def fit_strength(w, family, starts=None):
    models = []
    for i in range(3):
        x = VOICE_X if i==1 else X[:,:8]
        y = Q[:,i]
        use, active = w>0, (w>0)&(y>0)
        sd = np.r_[STRENGTH_PRIOR,.75] if i==1 else STRENGTH_PRIOR
        start_gate,start_amount = (None,None) if starts is None else starts[i]
        gate = logistic(x[use],y[use]>0,w[use],sd,start=start_gate)
        if i==0:
            amount = logistic(x[active],y[active]/W,w[active],sd,start=start_amount)
        elif i==1:
            amount = fit_positive_count(x[active],y[active],w[active],family,start_amount)
        else:
            amount = np.average(y[active],weights=w[active])
        models.append((gate,amount))
    return models


def predict_strength(x, fits, family, baseline):
    values = []
    for i,(gate,amount) in enumerate(fits):
        xi = np.concatenate([x,np.full((*x.shape[:-1],1),np.log1p(baseline))],axis=-1) if i==1 else x
        p = expit(xi@gate)
        if i==0:
            positive = W*expit(xi@amount)
        elif i==1:
            positive = count_moments(xi,amount,family)
        else:
            positive = np.full(p.shape,amount)
        mass = p@CONTEXT_WEIGHT
        expectation = (p*positive)@CONTEXT_WEIGHT
        values.append(np.stack([mass,expectation/mass,expectation],axis=-1))
    return np.stack(values,axis=1)


def summarize_times(p,time_x,time_models):
    # p is ordered Any, H, V, A, while time models are H, V, A, Any.
    p = p[..., [1,2,3,0]] * CONTEXT_WEIGHT[None,:,None]
    total = p.sum(axis=1)
    values = [time_parameters(time_x[...,terms],coef) for terms,coef in time_models]
    mu = np.stack([v[0] for v in values],axis=-1)
    sigma = np.array([v[1] for v in values])
    upper = (np.log(W)-mu)/sigma
    log_end = log_ndtr(upper)
    mean = np.exp(mu+.5*sigma**2+log_ndtr(upper-sigma)-log_end)
    lo, hi = np.zeros_like(total),np.full_like(total,W)
    for _ in range(28):
        t = (lo+hi)/2
        cdf = np.exp(log_ndtr((np.log(t[:,None,:])-mu)/sigma)-log_end)
        early = (p*cdf).sum(axis=1)<total/2
        lo,hi = np.where(early,t,lo),np.where(early,hi,t)
    return total,(lo+hi)/2,W*(1-total)+(p*mean).sum(axis=1)


def predictions(grids,occurrence,time_models,strength,family,baseline):
    standard,factors,scenario,time_x = grids
    combos = mix_predict(standard,occurrence).mean(axis=1)
    p = marginals(combos)
    f = {key:np.array([marginals(mix_predict(v,occurrence)).mean(axis=1) for v in group])
         for key,group in factors.items()}
    changes = []
    for sub in (f["contexts"][1:]-f["contexts"][:1],
                f["distance"][1:]-f["distance"][:1], f["size"][1:]-f["size"][:1]):
        natural,control = sub[:,1:].mean(axis=1),sub[:,0]
        changes.extend(np.stack([control,natural,natural-control],axis=1))
    window,median,rmst = summarize_times(marginals(mix_predict(scenario,occurrence)),time_x,time_models)
    any_medians = median[:,-1].reshape(4,3)
    return {"standardized_combos":combos,"standardized_response":p,
            "combo_contrasts":compare(combos),"response_contrasts":compare(p),
            "what_changes":np.asarray(changes),
            "context_gaps":f["contexts"][:,1:].mean(axis=1)-f["contexts"][:,0],
            "scenario_p":window,"scenario_median":median,"scenario_rmst":rmst,
            "any_change_50_to_150":any_medians[:,2]-any_medians[:,0],
            "strength":predict_strength(time_x[...,:8],strength,family,baseline)}


def paired_benchmarks():
    natural,control = YCODE[NAT]>0,YCODE[CON]>0
    plus,minus = np.sum(natural&~control),np.sum(~natural&control)
    delta = np.eye(8)[YCODE[NAT]]-np.eye(8)[YCODE[CON]]
    stat = abs(delta.sum(axis=0)).sum()/(2*len(NAT))
    rng = np.random.default_rng(341674)
    upper, repeats = 0, 99999
    for start in range(0,repeats,1000):
        k = min(repeats-start,1000)
        signs = 2*rng.integers(0,2,size=(k,len(delta)))-1
        upper += (np.abs(signs@delta).sum(axis=1)/(2*len(NAT))>=stat).sum()
    return {"response_natural":natural.sum(),"response_control":control.sum(),
            "only_natural":plus,"only_control":minus,"pairs":len(NAT),
            "mcnemar_p":binomtest(int(plus),int(plus+minus),.5).pvalue,
            "combination_tv":stat,"combination_permutation_p":(upper+1)/(repeats+1),
            "combination_permutations":repeats,
            "combination_counts":[np.bincount(YCODE[CTYPE==c],minlength=8) for c in range(4)]}


def reference_results(archive, occurrence, selection, strength, size, baseline):
    if archive is None:
        return None
    with ZipFile(archive) as source:
        def read(name):
            return list(csv.DictReader(io.StringIO(source.read("problem3/results/"+name).decode("utf-8-sig"))))

        raw = [r for r in read("r_baseline_predictions.csv")
               if (r["CallType"],float(r["Distance"]),r["Context"],float(r["Npenguins"]))
                  == ("flat",100.0,"searching",size)]
        x = design(np.array([1]),1,100,size)[:,:8]
        vocal_x = np.column_stack([x,np.log1p([baseline])])
        new_x = [features(np.array([1]),1,100,size,s["structure"]) for s in selection]
        response = marginals(mix_predict(new_x,tuple(map(np.asarray,occurrence))))[0]
        fitted = {
            "HeadsUp":expit(x@np.asarray(strength[0][0]))[0],
            "Vocal":expit(vocal_x@np.asarray(strength[1][0]))[0],
            "Approach":response[3],
        }
        assert {r["behavior"] for r in raw}==set(fitted)
        return {"r_diagnostics":read("r_baseline_diagnostics.csv"),
                "published_parameter_checks":read("q3_supplementary3_parameters.csv"),
                "benchmark":{"rows":raw,"merged_point":fitted,"baseline_calls":baseline}}


def calculate(repeats, archive):
    select,specs,cv = select_occurrence()
    timing = time_selection()
    family,count_cv = select_count()
    print(f"Occurrence: {select}; timings {[x['scale'] for x in timing]}; counts {family}",flush=True)
    size,baseline = float(np.median(N)),float(np.median(PRE))
    grids = prepare_grids(select,size)
    w = np.ones(len(N))
    occ = fit_occurrence(w,specs)
    strength = fit_strength(w,family)

    def fit_times(weight,starts=(None,)*4):
        return [(t["terms"],fit_endpoint(t["terms"],TIMES[:,i],weight,MIDPOINTS[i],
                                        t["scale"],start=starts[i]))
                for i,t in enumerate(timing)]

    times = fit_times(w)
    point = predictions(grids,occ,times,strength,family,baseline)
    samples = {key:[] for key in point}
    for i,weight in enumerate(track_bootstrap(repeats),1):
        draws = predictions(grids,fit_occurrence(weight,specs,occ),
                            fit_times(weight,[v for _,v in times]),fit_strength(weight,family,strength),
                            family,baseline)
        for key,v in draws.items():
            samples[key].append(v)
        if i%500==0:
            print(f"Completed {i}/{repeats} track resamples",flush=True)
    result = {key:summarize(value,samples[key]) for key,value in point.items()}
    for key in ("response_contrasts","combo_contrasts"):
        for a,b in ((0,4),(4,7)):
            result[key][a:b] = simultaneous(point[key][a:b],np.asarray(samples[key])[:,a:b])
    result["what_changes"] = simultaneous(point["what_changes"],samples["what_changes"])
    result["context_gaps"] = simultaneous(point["context_gaps"],samples["context_gaps"])
    np.testing.assert_allclose(point["standardized_combos"].sum(axis=-1),1)
    np.testing.assert_allclose(point["standardized_response"][:,0],1-point["standardized_combos"][:,0])
    np.testing.assert_allclose(point["strength"][...,0]*point["strength"][...,1],point["strength"][...,2])
    scenarios = []
    for i,(call,distance) in enumerate(CONDITIONS):
        detail = {}
        for j,b in enumerate(ENDPOINTS):
            support,status = observed_support(call,distance,size,j)
            whole_call_events = EVENTS[CTYPE==call,j].sum()
            applicable = whole_call_events>0
            detail[b] = {**support,"status":status,
                         "global_sound_events":whole_call_events,"p":result["scenario_p"][i,j],
                         "median":result["scenario_median"][i,j] if applicable else None,
                         "median_main":result["scenario_median"][i,j] if applicable and not status.startswith("no_response") else None,
                         "rmst":result["scenario_rmst"][i,j] if applicable else None}
        scenarios.append({"sound":CALLS[call],"distance":distance,"response":detail,"strength":result["strength"][i]})
    return {"experiment":paired_benchmarks(),"observed_subgroups":observed(),"repeats":repeats,
            "occurrence_selection":select,"occurrence_cv":cv,"occurrence_coefficients":occ,
            "timing_selection":timing,"timing_coefficients":times,"strength_coefficients":strength,
            "count_family":family,"count_cv":count_cv,"context_weights":dict(zip(CONTEXTS,CONTEXT_WEIGHT)),
            "strength_positive_observed":(Q>0).sum(axis=0),"strength_baseline":baseline,
            "group_size":size,"combo_labels":COMBOS,"response_labels":("Any",*BEHAVIORS),
            "contrast_labels":CONTRASTS,"context_support_n":CONTEXT_MASK.sum(),
            "factor_labels":("searching-commuting","after-commuting","100m-50m","8penguins-4penguins"),
            "factor_groups":("control","natural_equal","difference_of_differences"),
            "scenarios":scenarios,"estimates":result,
            "code1_reference":reference_results(archive,occ,select,strength,size,baseline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--bootstrap",type=int,default=3000)
    parser.add_argument("--teammate-zip",type=Path)
    args = parser.parse_args()
    if args.bootstrap<2:
        parser.error("--bootstrap must be at least 2")
    result = calculate(args.bootstrap,args.teammate_zip)
    args.output.write_text(json.dumps(result,default=serialize,ensure_ascii=False,
                                      allow_nan=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
