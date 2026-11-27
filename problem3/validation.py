from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, log_ndtr, logsumexp, ndtr
from scipy.stats import binomtest, chi2

from models import (BEHAVIORS, BITS, CALLS, COMBOS, WINDOW, count_fit, count_nll,
                    design, estimate, fit, logistic, predict)


def folds(d, seed):
    ids = np.random.default_rng(seed).permutation(d.expID.unique())
    return [d.expID.isin(part).to_numpy() for part in np.array_split(ids,5)]


def select(d, splits):
    scores = []
    for fold, test in enumerate(splits):
        for name in ["additive","pooled_interaction","full_interaction","quadratic"]:
            x = design(d,name)
            y = d.ResponseYesNo.to_numpy()
            b = logistic(x[~test],y[~test],np.ones((~test).sum()))
            eta = x[test] @ b
            scores.append(("overall",name,fold,test.sum(),
                           np.mean(np.logaddexp(0,eta)-y[test]*eta)))
        x = np.column_stack([design(d),np.log1p(d.VocalNcallsBefore)])
        y = d.VocalNcalls.to_numpy()
        tr,te = (~test)&(y>0),test&(y>0)
        for name in ["poisson","nb"]:
            b = count_fit(x[tr],y[tr],np.ones(tr.sum()),name)
            scores.append(("count",name,fold,te.sum(),count_nll(x[te],y[te],b,name)[0].mean()))
    rows = pd.DataFrame(scores,columns=["endpoint","model","fold","n","validation_nll"])
    def winner(a):
        loss = (a.validation_nll*a.n).groupby(a.model).sum()/a.n.groupby(a.model).sum()
        return loss.idxmin()
    return rows, winner(rows[rows.endpoint=="overall"]), winner(rows[rows.endpoint=="count"])


def aft_loss(x,t,event,b,family):
    mu, scale = x @ b[:-1], np.exp(b[-1])
    time = np.where(event, np.where(t==0,.5,t), WINDOW)
    z = (np.log(time)-mu)/scale
    if family=="lognormal":
        logpdf = -.5*z**2 - .5*np.log(2*np.pi) - b[-1] - np.log(time)
        logsurv, logcdf = log_ndtr(-z),log_ndtr(z)
    else:
        hazard = np.exp(np.clip(z,-100,45))
        logpdf = z - hazard - b[-1] - np.log(time)
        logsurv, logcdf = -hazard,np.log(-np.expm1(-hazard))
    return -np.where(event,np.where(t==0,logcdf,logpdf),logsurv)


def cross_validate(d,splits,structure,count_family):
    parts, comparisons = [], []
    for fold, test in enumerate(splits):
        train, val = d[~test],d[test]
        m = fit(train,np.ones(len(train)),structure,count_family)
        p, cp, _, time, intensity = predict(val,m,structure,count_family)
        result = val[["expID","Type","ResponseYesNo","code"]].copy()
        result["fold"], result["overall_probability"] = fold,p
        result["combo_logloss"] = -np.log(cp[np.arange(len(val)),val.code])
        # Positive durations are recorded as integer seconds. Benchmark the
        # original study's count family with the same marginal covariates.
        pos_head = train.HeadsUpDuration>0
        hx = design(train.loc[pos_head])
        hb = count_fit(hx,train.loc[pos_head,"HeadsUpDuration"].to_numpy(),np.ones(len(hx)),"nb")
        result["heads_positive_mean_nb"] = count_nll(design(val),1,hb,"nb")[1]
        result["heads_positive_mean_bounded"] = intensity[:,0,1]
        for j,(h,col) in enumerate(zip(BEHAVIORS,["HeadsUpDuration","VocalNcalls","ApproachDuration"])):
            pr, _, _, _, mu, sigma = time[:,j].T
            e, t = val[h+"YesNo"].to_numpy(bool),val[h+"Latency"].to_numpy()
            for horizon in [5,15,60]:
                pred = pr*ndtr((np.log(horizon)-mu)/sigma)/ndtr((np.log(60)-mu)/sigma)
                result[h+f"_observed_{horizon}"] = (e & (t<=horizon)).astype(int)
                result[h+f"_predicted_{horizon}"] = pred
            z = (np.log(np.where(e & (t>0),t,.5))-mu)/sigma
            logf = -.5*z*z - .5*np.log(2*np.pi)-np.log(sigma)-np.log(np.where(e & (t>0),t,.5))
            logf = np.where(t==0,log_ndtr(z),logf)-log_ndtr((np.log(60)-mu)/sigma)
            ll = np.where(e,np.log(pr)+logf,np.log1p(-pr))
            comparisons.append((h,"window_two_part",fold,len(val),-ll.mean()))
            x = design(train)
            et, tt = train[h+"YesNo"].to_numpy(bool),train[h+"Latency"].to_numpy()
            for family in ["lognormal","weibull"]:
                def loss(b):
                    return aft_loss(x,tt,et,b,family).sum()+.5*np.sum(b[1:-1]**2)
                b = estimate(loss,np.r_[np.log(60),np.zeros(x.shape[1]-1),1.],jac=False,
                             bounds=[(-12,20)]*x.shape[1]+[(-2.,3.)])
                nll = aft_loss(design(val),t,e,b,family)
                comparisons.append((h,family,fold,len(val),nll.mean()))
            result[col+"_observed"] = val[col]
            result[col+"_expected"] = intensity[:,j,2]
            result[col+"_p_positive"] = intensity[:,j,0]
        parts.append(result)
    return pd.concat(parts).sort_index(), pd.DataFrame(comparisons,columns=[
        "behavior","model","fold","n","validation_nll"])


def paired_tests(d, seed):
    wide = d.pivot(index="expID",columns="Type",values="ResponseYesNo")
    plus = ((wide.call==1)&(wide.control==0)).sum()
    minus = ((wide.call==0)&(wide.control==1)).sum()
    c = d.pivot(index="expID",columns="Type",values="code")
    differences = np.eye(8)[c.call]-np.eye(8)[c.control]
    stat = (differences.sum(axis=0)**2).sum()
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(100):
        signs = rng.integers(0,2,size=(500,len(c)))*2-1
        exceed += np.count_nonzero(((signs@differences)**2).sum(axis=1)>=stat)
    return int(plus),int(minus),binomtest(plus,plus+minus).pvalue,(1+exceed)/50001


def overall_checks(d,structure):
    # Score test of the six additional interactions under the unpenalized null.
    x = design(d)
    y = d.ResponseYesNo.to_numpy()
    b = logistic(x,y,np.ones(len(d)),0.)
    p = expit(x@b)
    interaction = design(d,"full_interaction")[:,x.shape[1]:]
    weights = (p*(1-p))[:,None]
    residual = interaction-x@np.linalg.solve(x.T@(weights*x),x.T@(weights*interaction))
    score = pd.DataFrame(residual*(y-p)[:,None]).groupby(d.expID.to_numpy()).sum().to_numpy()
    total = score.sum(axis=0)
    stat = total @ np.linalg.pinv(score.T@score) @ total
    # Integrate a shared normal intercept over each experiment with GH quadrature.
    grouped = d.sort_values(["expID","Type"])
    x = design(grouped,structure)
    y = grouped.ResponseYesNo.to_numpy().reshape(-1,2,1)
    nodes,w = np.polynomial.hermite.hermgauss(24)
    def loss(v):
        eta = (x@v[:-1]).reshape(-1,2,1)+np.sqrt(2)*nodes*np.exp(v[-1])
        ll = (y*eta-np.logaddexp(0,eta)).sum(axis=1)
        return -logsumexp(ll+np.log(w/np.sqrt(np.pi)),axis=1).sum()+.125*np.sum(v[1:-1]**2)
    b = estimate(loss,np.r_[logistic(x,y.ravel(),np.ones(len(x))),0.],jac=False,
                 bounds=[(-20,20)]*x.shape[1]+[(-8,3)])
    preds = []
    for c in CALLS:
        mat = design(d.assign(CallType=c),structure)
        eta = (mat@b[:-1])[:,None]+np.sqrt(2)*nodes*np.exp(b[-1])
        preds.append((expit(eta) @ (w/np.sqrt(np.pi))).mean())
    return chi2.sf(stat,6), np.exp(b[-1]), preds


def comparisons(p,combo):
    # Last slot denotes a three-call equal-weight mixture, standardized over
    # the exact same distance, group size and context rows as the control.
    p = np.r_[p,p[1:].mean()]
    combo = np.vstack([combo,combo[1:].mean(axis=0)])
    rows = []
    for a,b in [*[(j,0) for j in (1,2,3,4)],*combinations([1,2,3],2)]:
        rows.append(np.r_[p[a]-p[b], (p[a]/(1-p[a]))/(p[b]/(1-p[b])),combo[a]-combo[b]])
    return np.array(rows)
