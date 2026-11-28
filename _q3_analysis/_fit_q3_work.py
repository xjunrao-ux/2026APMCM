from collections import Counter
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logsumexp, log_ndtr, ndtr, ndtri, gammaln, digamma
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
W = 60.0
CALLS = ('control', 'flat', 'mod', 'tv')
CONTEXTS = ('commuting', 'searching', 'after')
BEHAVIORS = ('HeadsUp', 'Vocal', 'Approach')
INTENSITY = ('HeadsUpDuration', 'VocalNcalls', 'ApproachDuration')
MASKS = np.array([4, 2, 1])
INDICATORS = (np.arange(8)[:, None] & MASKS) > 0
SUBTYPE = np.array([[0, 0], [1/np.sqrt(2), 1/np.sqrt(6)],
                    [-1/np.sqrt(2), 1/np.sqrt(6)], [0, -2/np.sqrt(6)]])
OCC_SD = np.array([3, 2.5, 1, 1, 1.25, 1.25, .8, .8, 1, 1, .65])
COMBO_SD = np.array([1.8, 1.6, .8, .8, .8, .8, .55, .55, .65, .65, .55])
TIME_SD = np.array([2, .8, .5, .5, .5, .5, .45, .45])
STR_SD = np.array([1.7, .8, .5, .5, .5, .5, .4, .4])
DATA = pd.read_csv(ROOT / '题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv')
DATA = DATA.sort_values(['expID', 'Type']).reset_index(drop=True)
CTYPE = DATA.CallType.map(dict(zip(CALLS, range(4)))).to_numpy()
CTX = DATA.Context.map(dict(zip(CONTEXTS, range(3)))).to_numpy()
D = DATA.Distance.to_numpy()
N = DATA.Npenguins.to_numpy()
Y = DATA[[b + 'YesNo' for b in BEHAVIORS]].to_numpy(int)
YCODE = Y @ MASKS
YANY = (YCODE > 0).astype(int)
LAT = DATA[[b + 'Latency' for b in BEHAVIORS]].to_numpy(float)
FIRST = np.min(np.where(Y > 0, LAT, np.inf), axis=1)
Q = DATA[list(INTENSITY)].to_numpy(float)
PRE = DATA.VocalNcallsBefore.to_numpy()
NAT = np.where(CTYPE > 0)[0]
CON = np.where(CTYPE == 0)[0]
PAIRS = np.column_stack([NAT, CON])
SAME = D[NAT] == D[CON]
TRACK = DATA.PlaybackTrack.to_numpy()[NAT]
DAY = DATA.Date.to_numpy()[NAT]
TRACKS = [np.unique(TRACK[CTYPE[NAT] == c]) for c in (1, 2, 3)]
TRACK_ID = {t: np.where(TRACK == t)[0] for t in np.unique(TRACK)}
CONTEXT_WEIGHT = np.bincount(CTX[NAT]) / len(NAT)

assert np.array_equal(DATA.expID.to_numpy()[NAT], DATA.expID.to_numpy()[CON])
assert np.array_equal(YANY, DATA.ResponseYesNo.to_numpy())
# The two playbacks share a context, but observed group size may change.
assert np.array_equal(CTX[NAT], CTX[CON])
assert np.all(np.isfinite(LAT[Y > 0])) and np.all((LAT[Y > 0] >= 0) & (LAT[Y > 0] < W))


def design(call, context, distance, size):
    call, context, distance, size = np.broadcast_arrays(call, context, distance, size)
    natural = (call > 0).astype(float)
    after, search = context == 2, context == 1
    dist = np.log2(distance / 80)
    return np.column_stack([np.ones(len(call)), natural, SUBTYPE[call], after, search,
                            dist, np.log2(size / 8), natural*after, natural*search, natural*dist])


X = design(CTYPE, CTX, D, N)
SCENARIOS = np.array([(c, d) for c in range(4) for d in (50, 100, 150)])
XP = design(np.repeat(SCENARIOS[:, 0], 3), np.tile(range(3), 12),
            np.repeat(SCENARIOS[:, 1], 3), 8)
XNAT = np.array([design(c, CTX[NAT], D[NAT], N[NAT]) for c in range(4)])


def optimize(fun, center, sd, start=None, bounds=None):
    shape = center.shape
    def objective(z):
        z = z.reshape(shape)
        loss, grad = fun(z)
        d = (z - center) / sd
        return loss + .5*np.sum(d*d), (grad + d/sd).ravel()
    result = minimize(objective, (center if start is None else start).ravel(), jac=True,
                      bounds=bounds, method='L-BFGS-B', options={'ftol': 2e-11, 'gtol': 2e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    return result.x.reshape(shape)


def logistic(x, y, w, sd, center=None, start=None):
    if center is None:
        center = np.zeros(x.shape[1])
    def objective(coef):
        eta = x @ coef
        return np.dot(w, np.logaddexp(0, eta) - y*eta), x.T @ (w*(expit(eta)-y))
    return optimize(objective, center, sd, start)


def cloglog(x, y, w, sd, offset=None):
    off = np.zeros(len(y)) if offset is None else offset
    def objective(coef):
        eta = x @ coef + off
        rate = np.exp(np.clip(eta, -30, 20))
        ll = np.where(y, np.log(-np.expm1(-rate)), -rate)
        grad = np.where(y, rate*np.exp(-rate)/-np.expm1(-rate), -rate)
        return -np.dot(w, ll), -(x.T @ (w*grad))
    return optimize(objective, np.zeros(x.shape[1]), sd)


def softmax_fit(x, y, w, sd, start=None):
    ind = np.eye(7)[y-1]
    def objective(coef):
        eta = x @ coef
        z = logsumexp(eta, axis=1)
        prob = np.exp(eta - z[:, None])
        return np.dot(w, z - eta[np.arange(len(y)), y-1]), x.T @ (w[:, None]*(prob-ind))
    return optimize(objective, np.zeros((x.shape[1], 7)), sd[:, None], start)


def prob(x, occurrence):
    b, c = occurrence
    pa = expit(x @ b)
    z = x @ c
    pc = np.exp(z - logsumexp(z, axis=-1, keepdims=True))
    return np.concatenate([(1-pa)[..., None], pa[..., None]*pc], axis=-1)


def fit_occ(w, xc=X, oc_sd=OCC_SD, co_sd=COMBO_SD, codes=YCODE, start=None):
    ok = codes > 0
    a0, c0 = (None, None) if start is None else start
    a = logistic(xc, ok, w, oc_sd, start=a0)
    c = softmax_fit(xc[ok], codes[ok], w[ok], co_sd, start=c0)
    return a, c


def fit_time(x, time, w, midpoint, zero_upper=.5, start=None, coefficient_sd=TIME_SD):
    zero = time == 0
    center = np.zeros(x.shape[1] + 1)
    center[[0, -1]] = np.log([midpoint, .8])
    sd = np.r_[coefficient_sd, .6]
    lt = np.log(np.where(zero, zero_upper, time))
    def objective(coef):
        mu, sigma = x @ coef[:-1], np.exp(coef[-1])
        z = (lt-mu)/sigma
        zw = (np.log(W)-mu)/sigma
        cdf = log_ndtr(zw)
        lnum = np.where(zero, log_ndtr(z), -.5*z*z - np.log(sigma) - lt - .5*np.log(2*np.pi))
        dz = np.where(zero, np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z)), -z)
        dw = np.exp(-.5*zw*zw-.5*np.log(2*np.pi)-cdf)
        dmu = (dw-dz) / sigma
        dscale = -dz*z - (~zero).astype(float) + dw*zw
        return -np.dot(w, lnum-cdf), -np.r_[x.T @ (w*dmu), np.dot(w, dscale)]
    return optimize(objective, center, sd, start,
                    bounds=[(None, None)] * x.shape[1] + [(np.log(.14), np.log(2.5))])


def fit_count(x, y, w, start=None):
    center = np.zeros(x.shape[1]+1)
    center[[0, -1]] = np.log([3, 3])
    sd = np.r_[STR_SD, .45, 1.15]
    def objective(coef):
        eta = x @ coef[:-1]
        mu, r = np.exp(eta), np.exp(coef[-1])
        denom = r+mu
        ratio = -np.log1p(mu/r)
        lp0 = r*ratio
        positive = -np.expm1(lp0)
        ll = gammaln(y+r)-gammaln(r)-gammaln(y+1)+lp0+y*(eta-np.log(denom)) - np.log(positive)
        factor = np.exp(lp0) / positive
        dmu = y-(r+y)*mu/denom + factor*(-r*mu/denom)
        dr = digamma(y+r)-digamma(r)+ratio+1-(r+y)/denom+factor*(ratio+1-r/denom)
        return -np.dot(w,ll), -np.r_[x.T @ (w*dmu), np.dot(w, dr*r)]
    return optimize(objective, center, sd, start,
                    bounds=[(None, None)] * x.shape[1] + [(-4, 6)])


def time_fit_set(w, start=None, zero_upper=.5, y=Y, first=FIRST):
    out = []
    for b, (t, mid) in enumerate(zip([LAT[:,0], LAT[:,1], LAT[:,2], first], [3, 5, 10, 3])):
        ix = y[:, b] > 0 if b < 3 else np.isfinite(first)
        out.append(fit_time(X[ix, :8], t[ix], w[ix], mid, zero_upper, None if start is None else start[b]))
    return out


def strength_fit_set(w, start=None):
    out = []
    for b in range(3):
        ix = Y[:, b] != 0
        t = np.log2(1 + LAT[ix, b])
        if b == 1:
            xi = np.column_stack([X[ix, :8], np.log2(1 + PRE[ix])])
            out.append(fit_count(xi, Q[ix, b], w[ix], None if start is None else start[b]))
        else:
            xi = np.column_stack([X[ix, :8], t - np.log2(4 if b == 0 else 11)])
            capacity = W-LAT[ix, b]
            frac = np.minimum(Q[ix, b], capacity) / capacity
            out.append(logistic(xi, frac, w[ix], np.r_[STR_SD, .45], start=None if start is None else start[b]))
    return out


def pred_strength(model):
    x = XP[[3,12,21,30], :8]
    x = np.column_stack([x, np.zeros(4)])
    h = 57*expit(x @ model[0])
    mu = np.exp(x @ model[1][:-1]); r = np.exp(model[1][-1])
    v = mu / -np.expm1(-r*np.log1p(mu/r))
    a = 50*expit(x @ model[2]); a[0] = np.nan
    return np.column_stack([h, v, a])


def pred_time(times, pp):
    p = np.concatenate([pp @ INDICATORS, (1-pp[:,0])[:,None]], axis=1).reshape(12,3,4)
    weighted = p*CONTEXT_WEIGHT[None,:,None]
    marginal = weighted.sum(axis=1)
    median = np.empty((12,4))
    rmst = np.empty_like(median)
    for b, coef in enumerate(times):
        mu = (XP[:, :8] @ coef[:-1]).reshape(12,3)
        s = np.exp(coef[-1])
        trunc = ndtr((np.log(W)-mu)/s)
        weights = weighted[:,:,b]
        lo, hi = np.full(12,-20.), np.full(12,np.log(W))
        for _ in range(28):
            mid = (lo+hi)*.5
            cum = np.sum(weights*ndtr((mid[:,None]-mu)/s)/trunc, axis=1)
            left = cum < .5*marginal[:,b]
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
        median[:,b] = np.exp((lo+hi)*.5)
        et = np.exp(mu+.5*s*s)*ndtr((np.log(W)-mu)/s - s)/trunc
        rmst[:,b] = W*(1-marginal[:,b])+np.sum(weights*et, axis=1)
    median[:3,2] = np.nan
    rmst[:3,2] = np.nan
    return marginal, median, rmst


def paired(w):
    discordant = YANY[NAT] != YANY[CON]
    x = (X[NAT] - X[CON])[discordant][:,[1,2,3,6]]
    y = YANY[NAT][discordant]
    return logistic(x, y, w[NAT][discordant], np.array([8, 4, 4, 4]))


def summaries(occ, times, strength, w):
    prop = (w[NAT] @ YANY[NAT] / w[NAT].sum(), w[CON] @ YANY[CON] / w[CON].sum())
    raw = np.r_[prop, prop[0]-prop[1], prop[0]/prop[1] if prop[1] else np.inf,
               (w[NAT] @ (YANY[NAT] > YANY[CON])+.5)/(w[NAT] @ (YANY[NAT] < YANY[CON])+.5)]
    pooled = prob(XNAT, occ).mean(axis=1)
    nat_weights = np.bincount(CTYPE[NAT])[1:]/len(NAT)
    pnat = nat_weights @ pooled[1:]
    std = np.vstack([pooled, pnat])
    effect = std - std[0]
    px = prob(XP, occ)
    out = dict(raw=raw, pairs=paired(w), occ=occ[0], standard=std, delta=effect,
               rawcombo=np.array([np.bincount(YCODE[ids],weights=w[ids],minlength=8)/w[ids].sum() for ids in (CON,NAT)]),
               stdbeh=std@INDICATORS, deltabeh=effect@INDICATORS,
               pxx=px.reshape(12,3,8), str=pred_strength(strength), nb_r=np.exp(strength[1][-1]),
               time_coef=np.stack(times), coeff_strength=np.array([strength[0][8], strength[1][8], strength[2][8]]))
    out['support_p'], out['median'], out['rmst'] = pred_time(times,px)
    scenarios = [(ctx, 80, 8) for ctx in range(3)] + [(0,d,8) for d in (50,100,150)] + [(0,80,n) for n in (4,8,16)]
    grid = np.array([prob(design(np.arange(4),c,d,n),occ) for c,d,n in scenarios])
    ga = 1-grid[:,:,0]
    out['gridp'] = ga
    out['gridbe'] = grid@INDICATORS
    out['griddelta'] = ga[:,1:]@nat_weights - ga[:,0]
    out['griddiff'] = np.array([out['griddelta'][1]-out['griddelta'][0],out['griddelta'][2]-out['griddelta'][0],
                                out['griddelta'][5]-out['griddelta'][3],out['griddelta'][8]-out['griddelta'][6]])
    out['nat_between'] = np.array([std[1]-std[2],std[1]-std[3],std[2]-std[3]])
    sx = prob(XP[[3,12,21,30]],occ)@INDICATORS
    out['str_total'] = sx*out['str']
    out['med_delta'] = out['median'][3:]-np.tile(out['median'][:3],(3,1))
    return out


def cv():
    quad = np.column_stack([X, X[:,6]**2, X[:,1]*X[:,6]**2])
    candidates = {
        'intercept':([0],X,OCC_SD,COMBO_SD),
        'type_only':([0,1,2,3],X,OCC_SD,COMBO_SD),
        'additive':(list(range(8)),X,OCC_SD,COMBO_SD),
        'context_interaction':(list(range(10)),X,OCC_SD,COMBO_SD),
        'main':(list(range(11)),X,OCC_SD,COMBO_SD),
        'quadratic':(list(range(13)),quad,np.r_[OCC_SD,.35,.25],np.r_[COMBO_SD,.35,.25]),
        'cloglog_free':(list(range(11)),X,OCC_SD,COMBO_SD),
        'cloglog_offset':([0,1,2,3,4,5,6,8,9,10],X,OCC_SD,COMBO_SD),
    }
    out = {}
    for model, (keep, xall, osd, csd) in candidates.items():
        row_loss = np.zeros((len(YCODE),4))
        for ids in TRACK_ID.values():
            j = PAIRS[ids].ravel(); w = np.ones(len(YCODE)); w[j]=0
            x = xall[:,keep]
            if model.startswith('cloglog'):
                off = np.log(N/8) if model.endswith('offset') else np.zeros(len(YCODE))
                b = cloglog(x,YANY,w,osd[keep],off)
                c = softmax_fit(X[YANY>0],YCODE[YANY>0],w[YANY>0],COMBO_SD)
                pa = -np.expm1(-np.exp(x[j] @ b + off[j]))
                cc = X[j] @ c
                pc = np.column_stack([1-pa, pa[:,None]*np.exp(cc-logsumexp(cc,axis=1,keepdims=True))])
            else:
                fit = fit_occ(w,x,osd[keep],csd[keep]); pc = prob(x[j],fit)
                pa = 1-pc[:,0]
            row_loss[j,0] = -np.log(np.clip(pc[np.arange(len(j)),YCODE[j]],1e-12,1))
            row_loss[j,1] = -np.log(np.clip(np.where(YANY[j],pa,1-pa),1e-12,1))
            row_loss[j,2] = (pa-YANY[j])**2
            row_loss[j,3] = pa
        out[model] = row_loss.mean(axis=0)
        if model == 'main':
            out['main_avg_by_type'] = [row_loss[CTYPE==c,3].mean() for c in range(4)]
            out['main_avg_binned'] = [[int(k.sum()),row_loss[k,3].mean(),YANY[k].mean()]
                                      for k in np.array_split(np.argsort(row_loss[:,3]),5)]
    return out


def approximate_ci(draws, point):
    return np.stack([point, *np.nanquantile(np.asarray(draws), [.025,.975], axis=0)], axis=-1)


def simultaneous(point, draws):
    draws = np.asarray(draws)
    sd = draws.std(axis=0, ddof=1)
    max_t = np.max(np.abs((draws-point) / np.maximum(sd,1e-10)).reshape(len(draws),-1), axis=1)
    delta = np.quantile(max_t,.95)*sd
    return np.stack([point, np.maximum(-1,point-delta), np.minimum(1,point+delta)], axis=-1)


def permutation(rng):
    diff = (np.eye(8)[YCODE[NAT]]-np.eye(8)[YCODE[CON]])
    denominator = np.bincount(YCODE,minlength=8)
    stat = np.sum(diff.sum(axis=0)**2/denominator)
    nr = 19999
    perm = rng.choice(np.array([-1,1]), size=(nr,len(NAT))) @ diff
    pos = np.count_nonzero(np.sum(perm**2/denominator,axis=1) >= stat-1e-10)
    return [stat,pos,(pos+1)/(nr+1)]


def audit():
    observed = []
    for c,d in SCENARIOS:
        i = (CTYPE==c)&(D==d)
        observed.append(np.r_[i.sum(),Y[i].sum(axis=0),YANY[i].sum()])
    counts = np.array([np.bincount(YCODE[CTYPE==i],minlength=8) for i in range(4)])
    context = [[int(((CTYPE==i)&(CTX==j)).sum()),int(YANY[(CTYPE==i)&(CTX==j)].sum())] for i in range(4) for j in range(3)]
    both = np.array([[np.sum((YANY[NAT]==i)&(YANY[CON]==j)) for j in (0,1)] for i in (0,1)])
    same_both = np.array([[np.sum(SAME&(YANY[NAT]==i)&(YANY[CON]==j)) for j in (0,1)] for i in (0,1)])
    intensity_observed = []
    for c in range(4):
        block = []
        for b in range(3):
            i=(CTYPE==c)&(Y[:,b]==1)
            q = Q[i,b] if b==1 else np.minimum(Q[i,b], W-LAT[i,b])
            block.append([int(i.sum()), np.median(LAT[i,b]), q.mean(), np.median(q)]) if i.any() else block.append([0,None,None,None])
        intensity_observed.append(block)
    return dict(counts=counts, contexts=context, table=both, same_table=same_both, exact_support=observed,
                track_counts=[len(s) for s in TRACKS], track_n=[min(map(len,[TRACK_ID[t] for t in TRACK_ID])),max(map(len,[TRACK_ID[t] for t in TRACK_ID]))],
                observations_by_context=CONTEXT_WEIGHT, events_observed=intensity_observed,
                mcnemar=binomtest(both[1,0],both[1,0]+both[0,1],.5).pvalue,
                support_long=[np.bincount(CTYPE[D>=150],minlength=4),np.bincount(CTYPE[(D>=150)&(YANY>0)],minlength=4)],
                closest=np.min(D),furthest=np.max(D), distance_pair=np.median(D[NAT]-D[CON]),
                long_distance=np.bincount(CTYPE[D>=100],minlength=4))


def analyze(n_boot):
    w = np.ones(len(YCODE))
    occ = fit_occ(w)
    times = time_fit_set(w)
    intensity = strength_fit_set(w)
    p = summaries(occ,times,intensity,w)
    rng = np.random.default_rng(277311)
    boot = []
    coefficients = []
    fit_clog = cloglog(X,YANY,w,OCC_SD)
    for i in range(n_boot):
        tracks = [t for pool in TRACKS for t in rng.choice(pool,size=len(pool),replace=True)]
        counts = Counter(tracks)
        w = np.repeat([counts[t] for t in TRACK],2).astype(float)
        assert np.array_equal(w[NAT],w[CON])
        fit = fit_occ(w,start=occ)
        tb = time_fit_set(w,start=times)
        sb = strength_fit_set(w,start=intensity)
        boot.append(summaries(fit,tb,sb,w))
        coefficients.append(cloglog(X,YANY,w,OCC_SD)[7]/np.log(2))
    results = {k: approximate_ci([b[k] for b in boot],v) for k,v in p.items()}
    results['adjusted_simul'] = simultaneous(p['delta'][1:4,1:], [b['delta'][1:4,1:] for b in boot])
    results['natural_simul'] = simultaneous(p['nat_between'], [b['nat_between'] for b in boot])
    results['vocal_background'] = dict()
    for b in (0,1):
        y = Y.copy()
        if b==1:
            y[:,1] = Q[:,1]>0
        codes = y@MASKS
        fit = fit_occ(np.ones(len(YCODE)),codes=codes)
        estimated = prob(XNAT,fit).mean(axis=1)
        results['vocal_background'][b] = dict(raw_any=[int((codes[ids]>0).sum()) for ids in (CON,NAT)],
                                               vocal_diff=(estimated@INDICATORS)[1:,1]-(estimated@INDICATORS)[0,1])
    w = np.ones(len(YCODE))
    for zero in (.25,1.):
        tb = time_fit_set(w,zero_upper=zero)
        results[f'zero{zero}'] = pred_time(tb,prob(XP,occ))[1]
    alternative = {}
    for key, x, os, cs in (('additive',X[:,:8],OCC_SD[:8],COMBO_SD[:8]),
                            ('less_shrink',X,OCC_SD*1.5,COMBO_SD*1.5),
                            ('more_shrink',X,OCC_SD*.67,COMBO_SD*.67)):
        fit = fit_occ(w,x,os,cs)
        alt = prob(XNAT[:,:,:x.shape[1]],fit).mean(axis=1)
        alternative[key] = dict(std_any=1-alt[:,0],delta=alt-alt[0])
    results['alternative'] = alternative
    results['cv'] = cv()
    results['cloglog_logN'] = approximate_ci(coefficients,fit_clog[7]/np.log(2))
    results['audit'] = audit()
    results['permutation'] = permutation(rng)
    results['boot_n'] = n_boot
    results['main_strength_coefficients'] = intensity
    results['main_times'] = times
    return results


def serialize(obj):
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    raise TypeError(type(obj))


if __name__ == '__main__':
    import sys
    result = analyze(int(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result,default=serialize),encoding='utf-8')
