"""Run from any directory with Python, NumPy, pandas, SciPy and matplotlib."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from itertools import product
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
DATA = ROOT / "题目与要求/date"
Q2 = ROOT / "problem2"
if not Q2.is_dir():
    Q2 = ROOT / "11.27 问题二"
Q3 = ROOT / "problem3"
sys.path.insert(0, str(Q3))
from models import CALLS, COMBOS, CONTEXTS, design, combo_design, logistic, multinomial

SEED, BOOT = 20261127, 499
DIST = np.arange(5, 201)
SIZES = [3, 8, 15]
CONDITIONS = CONTEXTS + ["standardized", "equal_weight"]
KEYS = ["CallType", "Context", "Npenguins", "Distance"]


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.9g")


def hull_slice(cell, size):
    """Integer distances inside the observed (distance, size) hull at a fixed size."""
    distance = cell.Distance.to_numpy()
    count = cell.Npenguins.to_numpy()
    # The two ends of a horizontal hull slice are observations or lie on
    # segments joining observations on opposite sides of the requested size.
    ends = list(distance[count == size])
    for left in np.flatnonzero(count < size):
        right = count > size
        fraction = (size - count[left]) / (count[right] - count[left])
        ends.extend(distance[left] + fraction * (distance[right] - distance[left]))
    if not ends:
        return np.zeros(len(DIST), dtype=bool)
    return (DIST >= min(ends) - 1e-9) & (DIST <= max(ends) + 1e-9)


def acoustic():
    gam = pd.read_csv(Q2 / "results/gam_predictions.csv").query("hold == 0")
    weights = pd.read_csv(Q2 / "gam_baseline/results/weights_comparison.csv").set_index("metric").critic
    means, errors, curves = [], [], []
    knots = np.array([4, 8, 16, 32, 64, 128, 256])
    for ci, call in enumerate(CALLS[1:]):
        cols, se = [], []
        for mi, metric in enumerate(weights.index):
            a = pd.read_csv(next(DATA.glob(f"Data{3+ci+3*mi:02}_*.csv")))
            assert [int(c.strip("m")) for c in a] == list(knots)
            cols.append(a.mean().values)
            se.append(np.interp(np.log2(DIST), np.log2(knots), a.std(ddof=0)/np.sqrt(a.count())))
        means.append(np.column_stack(cols))
        errors.append(np.sqrt(np.sum((weights.values[:, None]*se)**2, axis=0)))
        g = gam[gam.call == call].pivot(index="d",columns="metric",values="estimate")
        full = g[weights.index] @ weights
        curves.append(full.loc[DIST].values)
        refs = pd.read_csv(Q2 / "gam_baseline/results/gam_effective_range.csv")
        for row in refs[(refs.call==call) & refs.weight.str.startswith("critic")].itertuples():
            assert abs(full.index[full < row.threshold][0] - row.D) <= .101
    x = np.vstack(means)
    x = (x-x.min(0))/np.ptp(x,axis=0)
    info = x.std(0)*(1-np.corrcoef(x.T)).sum(1)
    np.testing.assert_allclose(info/info.sum(),weights)
    return np.array(curves), np.array(errors)


def unified_scenarios(environment, curves, radius, control_draws, delta_draws):
    """Representative distances; each column of bootstrap draws refers to the same expID sample."""
    keys = ["call", "context", "N", "distance"]
    model = curves.rename(columns=dict(zip(KEYS, keys)))
    env = environment.merge(model[keys + ["F", "control", "sim_lower95"]],
                            on=keys, validate="many_to_one")
    # Fixed recognition threshold for all evaluation standards. Also evaluate
    # harder recognition with tau=T, omitting the duplicate at T=0.5.
    harder_cases = env.loc[env["T"] > .5].copy()
    harder_cases["tau"] = harder_cases["T"]
    df = pd.concat([env.assign(tau=.5), harder_cases], ignore_index=True)
    harder = df.tau > .5
    degraded = df.loss > 0
    df["scenario"] = np.select(
        [harder & degraded, harder, degraded & (df["T"] > .5), degraded, df["T"] > .5],
        ["combined_stress", "harder_recognition", "loss_stricter_evaluation", "signal_loss",
         "stricter_evaluation"], default="baseline")
    a = (expit(20*(df.F_scenario-df.tau)) / expit(20*(df.F-.5))).to_numpy()
    assert np.all((a > 0) & (a <= 1))
    ci = df.call.map(CALLS[1:].index).to_numpy()
    cj = df.context.map(CONDITIONS.index).to_numpy()
    nk = df.N.map(SIZES.index).to_numpy()
    h = df.distance.to_numpy() - DIST[0]
    net_draws = a * delta_draws[:, ci, cj, nk, h]
    event_draws = control_draws[:, cj, nk, h] + net_draws
    df["recognition_retained"] = a
    df["delta_EI"] = a * df.delta_distance_model
    df["EI"] = df.control + df.delta_EI
    df["EI_lo95"], df["EI_hi95"] = np.quantile(event_draws, [.025, .975], axis=0)
    df["delta_lo95"], df["delta_hi95"] = np.quantile(net_draws, [.025, .975], axis=0)
    df["delta_sim_lower95"] = a * df.sim_lower95
    # At each distance a>0 is common to all replicates. The studentized
    # band scales by a, so its sign and the precomputed radius rule coincide.
    assert np.array_equal(df.delta_sim_lower95 > 0, df.sim_lower95 > 0)
    unchanged = ~harder
    np.testing.assert_allclose(df.EI[unchanged], df.EI_recognition_scenario[unchanged])
    np.testing.assert_allclose(df.EI[~harder & ~degraded], df.EI_distance_model[~harder & ~degraded])
    df = df.drop(columns=[
        "EI_recognition_scenario","delta_recognition_scenario","sim_lower95"])
    join = ["call","context","N","T","loss"]
    point_radius = radius[radius.F_shift_SE == 0]
    for scope in ("pooled_distance", "joint_hull"):
        subset = point_radius[point_radius.scope == scope]
        fields = ["start_m","limit_m","radius_m","first_limit"]
        df = df.merge(subset[join + fields].rename(columns={f:scope+"_"+f for f in fields}),
                      on=join, validate="many_to_one")
    estimates = []
    for label, rho in [("one_participant", 1/df.N), ("half_on_average", .5), ("all", 1.)]:
        size = df.N*rho
        estimates.append(df.assign(participation=label, rho_union=rho,
            expected_union=size*df.EI, count_lo95=size*df.EI_lo95, count_hi95=size*df.EI_hi95))
    result = pd.concat(estimates, ignore_index=True)
    for field in ["expected_union","count_lo95","count_hi95"]:
        assert result[field].between(0,result.N).all()
    assert len(result)==len(CALLS[1:])*len(CONDITIONS)*len(SIZES)*3*25*3
    assert not result.duplicated(keys+["T","tau","loss","participation"]).any()
    typical = result.query("N==8 and context=='standardized' and distance==100 and participation=='half_on_average'")
    baseline = typical[typical.scenario == "baseline"].set_index("call")
    np.testing.assert_array_equal(baseline.loc[CALLS[1:],"pooled_distance_radius_m"], [76,148,175])
    np.testing.assert_array_equal(baseline.loc["tv",["joint_hull_start_m","joint_hull_limit_m"]], [44,110])
    example = typical.query("call=='tv' and T==.5 and loss in [0,.2]").sort_values("loss")
    np.testing.assert_allclose(example.expected_union, [2.58,1.74], atol=.005)
    save(result, "unified_scenarios")


def design_evidence(d):
    """Count matches and changes once per trial pair in each natural-call stratum."""
    pairs = d[d.Type=="call"].merge(d[d.Type=="control"], on="expID",
        suffixes=("", "_control"), validate="one_to_one")
    pairs["distance_changed"] = pairs.Distance != pairs.Distance_control
    pairs["size_changed"] = pairs.Npenguins != pairs.Npenguins_control
    stats = dict(pairs=("expID","size"), natural_tracks=("PlaybackTrack","nunique"),
        control_tracks=("PlaybackTrack_control","nunique"), distance_min=("Distance","min"),
        distance_max=("Distance","max"), N_min=("Npenguins","min"), N_max=("Npenguins","max"),
        pairs_distance_changed=("distance_changed","sum"), pairs_size_changed=("size_changed","sum"))
    groups = pairs.groupby(["CallType","Context"]).agg(**stats).reset_index()
    total = pairs.assign(CallType="all",Context="all").groupby(["CallType","Context"]).agg(**stats)
    save(pd.concat([total.reset_index(), groups]), "design_evidence")


def run():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(next(DATA.glob("Data02*.csv")))
    pair = d.groupby("expID")
    assert (pd.crosstab(d.expID, d.Type)[["call","control"]] == 1).all().all()
    assert pair.Context.nunique().max() == 1
    natural = d[d.Type=="call"]
    np.testing.assert_equal(natural.Npenguins.quantile([.25,.5,.75]).values, SIZES)
    weights = natural.Context.value_counts(normalize=True).reindex(CONTEXTS).values
    code = d[["HeadsUpYesNo","VocalYesNo","ApproachYesNo"]].values @ [4,2,1]
    pos = code > 0
    assert np.array_equal(pos,d.ResponseYesNo)
    x,z = design(d),design(d,"full_interaction")
    grid = pd.DataFrame(product(CALLS, CONTEXTS, SIZES, DIST),columns=KEYS)
    gx,gz = combo_design(design(grid)), design(grid,"full_interaction")

    def predict(w):
        p = expit(gz @ logistic(z,pos,w))
        q = softmax(gx @ multinomial(x[pos],code[pos],w[pos],2.0),axis=1)
        pi = np.column_stack([1-p,p[:,None]*q])
        assert pi.min() >= 0 and np.allclose(pi.sum(1),1)
        raw = pi.reshape(len(CALLS),len(CONTEXTS),len(SIZES),len(DIST),8)
        avg = np.einsum("ckndj,k->cndj",raw,weights)
        return np.concatenate([raw,avg[:,None],raw.mean(axis=1,keepdims=True)],axis=1)

    pi = predict(np.ones(len(d)))
    full = pd.DataFrame(product(CALLS,CONDITIONS,SIZES,DIST),columns=KEYS)
    for j,combo in enumerate(COMBOS):
        full["p_"+combo] = pi[...,j].ravel()
    check = pd.read_csv(Q3 / "results/q3_behavior_combo_predictions.csv",dtype={"combo":str})
    ref = check.pivot(index=KEYS,columns="combo",values="p_all")
    aligned = full.set_index(KEYS).loc[ref.index,["p_"+s for s in ref.columns]]
    residual = np.max(np.abs(aligned.values-ref.values))
    assert residual < 1e-5, residual
    union = np.array([("A" in c or "V" in c) for c in COMBOS])
    ei = pi[...,union].sum(-1)
    for event in ("V","A","VA","H"):
        full["P_"+event] = pi[...,[all(b in c for b in event) for c in COMBOS]].sum(-1).ravel()
    np.testing.assert_allclose(full.P_V+full.P_A-full.P_VA,ei.ravel())
    full["EI"] = ei.ravel()
    save(full,"behavior")
    delta = ei[1:] - ei[0]
    samples = np.empty((BOOT,*delta.shape))
    control_samples = np.empty((BOOT,*ei[0].shape))
    ids,inv = np.unique(d.expID,return_inverse=True)
    rng = np.random.default_rng(SEED)
    for b in range(BOOT):
        w = np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[inv]
        pred = predict(w)[...,union].sum(-1)
        samples[b] = pred[1:]-pred[0]
        control_samples[b] = pred[0]
    lo,hi = np.quantile(samples,[.025,.975],axis=0)
    sd = samples.std(0,ddof=1)
    critical = np.quantile(np.max(np.abs((samples-delta)/sd),axis=-1),.95,axis=0)
    band = delta-critical[...,None]*sd
    F,se = acoustic()
    rows, radii, scenarios, stability = [], [], [], []
    joint_support = {(arm, context, n): hull_slice(cell, n)
                     for (arm, context), cell in d.groupby(["CallType", "Context"])
                     for n in SIZES}
    for i,call in enumerate(CALLS[1:]):
        for j,ctx in enumerate(CONDITIONS):
            contexts = [ctx] if ctx in CONTEXTS else CONTEXTS
            cells = d[d.CallType.isin([call,"control"]) & d.Context.isin(contexts)].groupby(["CallType","Context"])
            left,right = cells.Distance.min().max(),cells.Distance.max().min()
            for k,n in enumerate(SIZES):
                supported_n = bool((cells.Npenguins.min()<=n).all() and (cells.Npenguins.max()>=n).all())
                within_hull = np.logical_and.reduce([joint_support[arm, context, n]
                    for arm in (call, "control") for context in contexts])
                ix = i,j,k
                part = pd.DataFrame(dict(Distance=DIST,EI=ei[i+1,j,k],control=ei[0,j,k],delta=delta[ix],
                    lo95=lo[ix],hi95=hi[ix],sim_lower95=band[ix],sim_upper95=2*delta[ix]-band[ix],
                    F=F[i],F_SE=se[i],within_cell_ranges=(DIST>=left)&(DIST<=right)&supported_n,
                    within_joint_hull=within_hull))
                part = part.assign(CallType=call,Context=ctx,Npenguins=n)
                rows.append(part)
                for loss,t in product([0,.05,.10,.15,.20],[.5,.6,.7]):
                    f = (1-loss)*F[i]
                    # Optional recognition-mediated scenario, anchored to observed-condition EI.
                    # A higher evaluation threshold alone leaves the response law unchanged.
                    ratio = expit(20*(f-.5))/expit(20*(F[i]-.5))
                    for dist in [50,100,150]:
                        h = dist-DIST[0]
                        scenarios.append((call,ctx,n,dist,t,loss,f[h],f[h]>=t,part.EI[h],part.delta[h],
                            part.control[h]+ratio[h]*part.delta[h],ratio[h]*part.delta[h],
                            part.within_cell_ranges[h],within_hull[h]))
                    for scope,domain in (("pooled_distance",np.ones(len(DIST),bool)),
                                         ("cell_ranges",part.within_cell_ranges.values),
                                         ("joint_hull",within_hull)):
                        grid = DIST[domain]
                        if len(grid):
                            # Resample the underlying positive-effect boundary, separate from
                            # the confidence-band decision rule reported as the main radius.
                            ok = (samples[:,i,j,k,:][:,domain]>0) & (f[domain]>=t)
                            length = np.cumprod(ok,axis=1).sum(1)
                            nonempty = length>0
                            quant = (np.quantile(grid[length[nonempty]-1],[.025,.5,.975])
                                     if nonempty.any() else [np.nan]*3)
                            stability.append((call,ctx,n,t,loss,scope,nonempty.mean(),*quant))
                        for shift in [-1,0,1]:
                            good_f = ((1-loss)*(F[i]+shift*se[i])>=t)[domain]
                            good_b = band[ix][domain]>0
                            bad = np.where(~(good_f & good_b))[0]
                            end = bad[0] if len(bad) else len(grid)
                            why = ("unsupported" if not len(grid) else "support_limit" if not len(bad)
                                   else "+".join(label for ok,label in [(good_f[end],"F"),(good_b[end],"behavior")] if not ok))
                            radii.append((call,ctx,n,t,loss,scope,shift,grid[0] if len(grid) else None,
                                grid[-1] if len(grid) else None,grid[end-1] if end else None,why))
    curves = pd.concat(rows)
    save(curves,"effects")
    radius = pd.DataFrame(radii,columns=["call","context","N","T","loss","scope","F_shift_SE","start_m","limit_m","radius_m","first_limit"])
    save(radius,"radius")
    save(pd.DataFrame(stability,columns=["call","context","N","T","loss","scope","fraction_nonempty",
        "positive_radius_q025","positive_radius_q50","positive_radius_q975"]),"bootstrap_radius")
    environment = pd.DataFrame(scenarios,columns=["call","context","N","distance","T","loss","F_scenario","passes_T","EI_distance_model",
        "delta_distance_model","EI_recognition_scenario","delta_recognition_scenario","within_cell_ranges","within_joint_hull"])
    save(environment,"environment")
    unified_scenarios(environment, curves, radius, control_samples, samples)
    design_evidence(d)
    counts = full[full.Distance.isin([50,100,150])].copy()
    estimates=[]
    for label,rho in [("one_participant",1/counts.Npenguins),("half_on_average",.5),("all",1.)]:
        estimates.append(counts[KEYS].assign(scenario=label,rho=rho,
            expected_vocal=counts.Npenguins*rho*counts.P_V,expected_approach=counts.Npenguins*rho*counts.P_A,
            expected_union=counts.Npenguins*rho*counts.EI))
    save(pd.concat(estimates),"participant_scenarios")
    # Recover the target event in held-out trials using Q3's nested-CV folds and structures.
    oof = pd.read_csv(Q3/"results/grouped_oof.csv")
    oof = d[["expID","Type"]].merge(oof,on=["expID","Type"],validate="one_to_one")
    assert oof.groupby("expID").fold.nunique().max()==1
    cv = np.empty(len(d))
    for _,block in oof.groupby("fold"):
        test = oof.index.isin(block.index)
        train = ~test
        zfold = design(d,block.overall_structure.unique().item())
        gate = logistic(zfold[train],pos[train],np.ones(train.sum()))
        m = train & pos
        cl = multinomial(x[m],code[m],np.ones(m.sum()),2.0)
        cv[test] = expit(zfold[test]@gate)*softmax(combo_design(x[test])@cl,axis=1)[:,union[1:]].sum(1)
    observed = union[code]
    save(d[["expID","Type","CallType","Context"]].assign(fold=oof.fold,observed_EI=observed.astype(int),
        predicted_EI=cv,squared_error=(cv-observed)**2),"held_out_EI")
    calibration = pd.DataFrame(dict(predicted=cv,observed=observed))
    save(calibration.groupby(pd.qcut(cv,5,labels=False)).agg(n=("observed","size"),
        observed=("observed","mean"),predicted=("predicted","mean")).reset_index(names="bin"),"calibration")
    grp=curves.query("Context=='standardized' and Npenguins==8")
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True,layout="constrained")
    for ax,(call,g) in zip(axes,grp.groupby("CallType",sort=False)):
        ax.fill_between(g.Distance,g.sim_lower95,np.minimum(1,g.sim_upper95),alpha=.17,color="#226d79")
        ax.plot(g.Distance,g.delta,color="#226d79",label="Net response")
        ax.plot(g.Distance,g.F,color="#ab5828",label="Fidelity")
        ax.axhline(0,color="gray",lw=.8)
        ax.axhline(.5,color="#ab5828",lw=.8,ls=":")
        ax.set(title=call,xlabel="Playback distance (m)")
    axes[0].set_ylabel("Probability difference / fidelity")
    axes[-1].legend(frameon=False)
    fig.suptitle("8 penguins · Data02 context frequencies · 95% simultaneous band per curve")
    fig.savefig(OUT/"joint_curves.png",dpi=190)
    print("Q3 maximum probability discrepancy:",residual)
    print(radius.query("context=='standardized' and N==8 and loss==0 and F_shift_SE==0").to_string(index=False))
    print(grp.query("Distance==100").to_string(index=False))
    print("pair mismatches:",(pair[['Distance','Npenguins','Context']].nunique()>1).sum().to_dict())

if __name__=="__main__":
    run()
