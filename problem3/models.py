"""Estimators for one paired playback experiment per cluster."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_ndtr, logsumexp, ndtr, ndtri, softmax

CALLS = ["control", "flat", "mod", "tv"]
CONTEXTS = ["after", "commuting", "searching"]
BEHAVIORS = ["HeadsUp", "Vocal", "Approach"]
COMBOS = ["000", "A", "V", "VA", "H", "HA", "HV", "HVA"]
BITS = [4, 2, 1]
WINDOW = 60.


def design(d, structure="additive"):
    call = np.column_stack([d.CallType == c for c in CALLS[1:]])
    ctx = np.column_stack([d.Context == c for c in CONTEXTS[1:]])
    parts = [np.ones(len(d)), call, ctx, (d.Distance - 80) / 50, (d.Npenguins - 8) / 10]
    if structure == "pooled_interaction":
        parts.append(ctx * call.any(axis=1)[:, None])
    elif structure == "full_interaction":
        parts.extend([ctx * v[:, None] for v in call.T])
    elif structure == "quadratic":
        parts.extend([parts[-2] ** 2, parts[-1] ** 2])
    return np.column_stack(parts)


def estimate(objective, initial, jac=True, bounds=None):
    r = minimize(objective, initial, jac=jac, bounds=bounds, method="L-BFGS-B",
                 options={"ftol": 1e-11, "gtol": 2e-6, "maxiter": 400, "maxls": 60})
    if not r.success or not np.isfinite(r.fun):
        raise RuntimeError(str(r.message))
    return r.x


def logistic(x, y, w, penalty=.25):
    # Fractional outcomes also give a bounded conditional mean.
    reg = np.full(x.shape[1], float(penalty))
    reg[0] = .0001
    def objective(b):
        eta = x @ b
        return (w @ (np.logaddexp(0, eta) - y * eta) + .5 * (reg * b) @ b,
                x.T @ (w * (expit(eta) - y)) + reg * b)
    return estimate(objective, np.zeros(x.shape[1]))


def combo_design(x):
    # A shared natural-vs-control effect lets natural calls borrow information
    # from one another without shrinking all three directly to the control.
    return np.column_stack([x, x[:,1:4].sum(axis=1)])


def multinomial(x, y, w, penalty):
    x = combo_design(x)
    ref = np.zeros((14, x.shape[1]))
    ref[:, 0] = 1
    ref[7:,1:4] = 1/3
    ref[7:,-1] = 1
    x = np.vstack([x, ref])
    target = np.eye(7)[np.r_[y - 1, np.tile(np.arange(7),2)]]
    weights = np.r_[w, np.full(14, .25)]
    reg = np.full((x.shape[1], 1), penalty)
    reg[0] = 0.
    reg[-1] = .10 * penalty
    def objective(flat):
        b = flat.reshape(-1, 7)
        eta = x @ b
        loss = weights @ (logsumexp(eta, axis=1) - (eta * target).sum(axis=1))
        grad = x.T @ ((softmax(eta, axis=1) - target) * weights[:, None])
        return loss + .5 * (reg * b * b).sum(), (grad + reg * b).ravel()
    return estimate(objective, np.zeros(x.shape[1] * 7)).reshape(-1, 7)


def time_design(x, behavior):
    # No approach to control; only 23 approach events. Pool timing across
    # treatments, retaining the much better measured incidence differences.
    return x[:, :1] if behavior == "Approach" else x[:, :8]


def timing(x, t, w):
    # Lognormal conditional on T <= WINDOW. Immediate head raises carry
    # interval likelihood F(.5), not an invented positive event time.
    logt = np.log(np.where(t > 0, t, .5))
    left = t == 0
    reg = np.full(x.shape[1], 2.)
    reg[0] = .001
    c = .5 * np.log(2 * np.pi)
    def objective(b):
        mu, sigma = x @ b[:-1], np.exp(b[-1])
        z = (logt - mu) / sigma
        u = (np.log(WINDOW) - mu) / sigma
        mills_u = np.exp(-.5 * u*u - c - log_ndtr(u))
        nll = .5 * z*z + c + b[-1] + logt
        dm, ds = -z / sigma, 1 - z*z
        mills_z = np.exp(-.5 * z[left]**2 - c - log_ndtr(z[left]))
        nll[left] = -log_ndtr(z[left])
        dm[left], ds[left] = mills_z / sigma, mills_z * z[left]
        value = w @ (nll + log_ndtr(u)) + .5 * (reg * b[:-1]) @ b[:-1]
        grad = np.r_[x.T @ (w * (dm - mills_u / sigma)) + reg*b[:-1],
                     w @ (ds - u*mills_u)]
        return value, grad
    initial = np.r_[np.average(logt, weights=w), np.zeros(x.shape[1]-1), -.4]
    return estimate(objective, initial, bounds=[(None, None)]*x.shape[1]+[(-2.5, 1.)])


def time_summary(x, b, p):
    mu, sigma = x @ b[:-1], np.exp(b[-1])
    upper = (np.log(WINDOW)-mu)/sigma
    f = ndtr(upper)
    median = np.exp(mu + sigma * ndtri(.5 * f))
    mean = np.exp(mu + .5*sigma**2 + log_ndtr(upper - sigma) - log_ndtr(upper))
    return np.column_stack([p, 1-p, median, WINDOW*(1-p)+p*mean, mu,
                            np.full(len(x), sigma)])


def window_time_cdf(seconds, mu, sigma):
    """Fraction of within-window responders that have started by time t."""
    seconds = np.asarray(seconds, dtype=float)
    if np.any((seconds < 0) | (seconds > WINDOW)):
        raise ValueError("Time must lie inside the observation window")
    log_t = np.log(np.maximum(seconds, np.finfo(float).tiny))
    log_f = log_ndtr((log_t - mu) / sigma)
    log_f_window = log_ndtr((np.log(WINDOW) - mu) / sigma)
    return np.where(seconds == 0, 0., np.exp(log_f - log_f_window))


def count_nll(x, y, b, family):
    eta = np.clip(x @ b[:x.shape[1]], -12, 10)
    mean = np.exp(eta)
    if family == "poisson":
        logzero = -mean
        logpmf = y*eta - mean - gammaln(y+1)
    else:
        size = np.exp(b[-1])
        logzero = -size*np.log1p(mean/size)
        logpmf = (gammaln(y+size)-gammaln(size)-gammaln(y+1)
                  + logzero + y*(eta-np.log(size+mean)))
    return -logpmf + np.log(-np.expm1(logzero)), mean / -np.expm1(logzero)


def count_fit(x, y, w, family):
    n = x.shape[1]
    def loss(b):
        return w @ count_nll(x,y,b,family)[0] + .5 * np.sum(b[1:n]**2)
    extra = [(np.log(.05),np.log(200.))] if family == "nb" else []
    return estimate(loss, np.zeros(n+len(extra)), jac=False, bounds=[(-8,8)]*n+extra)


def fit(d, w, structure, count_family, combo_penalty=2., flag_from_time=False):
    x = design(d)
    z = design(d, structure)
    flags = np.column_stack([d[h+"YesNo"] for h in BEHAVIORS])
    if flag_from_time:
        flags = np.column_stack([d[h+"Latency"].notna() for h in BEHAVIORS])
    codes = flags @ BITS
    pos = codes > 0
    overall = logistic(z, d.ResponseYesNo.to_numpy(), w)
    gate = overall if np.array_equal(pos, d.ResponseYesNo) else logistic(z,pos,w)
    classes = multinomial(x[pos], codes[pos].astype(int), w[pos], combo_penalty)
    times, intensity = [], []
    for j,(h,col) in enumerate(zip(BEHAVIORS, ["HeadsUpDuration","VocalNcalls","ApproachDuration"])):
        mask = flags[:,j].astype(bool) & d[h+"Latency"].notna().to_numpy()
        times.append(timing(time_design(x[mask],h), d.loc[mask,h+"Latency"].to_numpy(),w[mask]))
        y = d[col].to_numpy()
        mask = y>0
        xi = np.column_stack([x,np.log1p(d.VocalNcallsBefore)]) if h=="Vocal" else x
        first = logistic(xi,mask,w,penalty=1.)
        if h == "Vocal":
            second = count_fit(xi[mask], y[mask], w[mask], count_family)
        elif h == "HeadsUp":
            second = logistic(xi[mask], y[mask]/WINDOW, w[mask],penalty=.4)
        else:
            second = np.average(y[mask],weights=w[mask])
        intensity.append((first,second))
    return overall, gate, classes, times, intensity


def predict(d, model, structure, count_family):
    overall, gate, classes, times, intensity = model
    x, z = design(d), design(d,structure)
    p = expit(z @ gate)
    q = softmax(combo_design(x) @ classes, axis=1)
    combos = np.column_stack([1-p, p[:,None]*q])
    latency, measures = [], []
    for j,h in enumerate(BEHAVIORS):
        incidence = combos[:,(np.arange(8) & BITS[j])>0].sum(axis=1)
        latency.append(time_summary(time_design(x,h),times[j],incidence))
        b, c = intensity[j]
        xi = np.column_stack([x,np.log1p(d.VocalNcallsBefore)]) if h=="Vocal" else x
        chance = expit(xi @ b)
        if h=="Vocal":
            amount = count_nll(xi,1,c,count_family)[1]
        elif h=="HeadsUp":
            amount = WINDOW*expit(xi @ c)
        else:
            amount = np.full(len(x),c)
        measures.append(np.column_stack([chance,amount,chance*amount]))
    return (expit(z @ overall), combos, q, np.stack(latency,axis=1),
            np.stack(measures,axis=1))
