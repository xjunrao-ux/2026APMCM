"""Build the Q1.4 evidence tables and report from the unpacked merged project."""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from common import COLORS, GROUPS, ROOT, holm
from evidence import BEHAVIORS, MEASURES, propagation_summary
import matplotlib.pyplot as plt


RAW = ROOT.parent / "2026APMCM_C_P1" / "01_数据"
OUT = ROOT.parent / "integrated" / "q1_4"
CLASS_TABLES = ROOT / "02_classification" / "tables"
OUTCOMES = dict(zip(BEHAVIORS, ("总体响应", "抬头", "鸣叫", "靠近")))
SEED = 20261127
SIMULATIONS = 200_000


def raw_csv(number):
    path, = RAW.glob(f"Data{number:02}_*.csv")
    return pd.read_csv(path)


def save(table, name):
    table.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def classification():
    keys = ["source_row", "CallType", "Date", "Location"]
    oof = pd.read_csv(CLASS_TABLES / "classification_oof.csv",
                      usecols=keys + ["fold", "blank6_stack"])
    # Fpeak in the older OOF export is a model label. Take all acoustics from Data01.
    calls = raw_csv(1).rename_axis("row").reset_index()
    calls["source_row"] = calls.pop("row") + 1
    calls["Date"] = pd.to_datetime(calls.Date, format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    calls = calls.merge(oof, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not calls["_merge"].eq("both").all() or not calls.blank6_stack.isin(GROUPS).all():
        raise ValueError("Data01 and the saved classifier predictions are not aligned.")
    matrix = pd.crosstab(calls.CallType, calls.blank6_stack).loc[list(GROUPS), list(GROUPS)]
    reference = pd.read_csv(CLASS_TABLES / "cascade_confusion.csv", index_col=0)
    np.testing.assert_array_equal(matrix, reference.loc[list(GROUPS), list(GROUPS)])
    by_type = calls.groupby("CallType")
    metrics = by_type.agg(call_n=("source_row", "size"), F02_measured=("F02", "count"),
                          pulse_measured=("PulseRate", "count"), Fpeak_median_Hz=("Fpeak", "median"))
    metrics["correct"] = np.diag(matrix)
    metrics["precision"] = metrics.correct / matrix.sum(axis=0).to_numpy()
    metrics["recall"] = metrics.correct / metrics.call_n
    metrics["F1"] = 2 * metrics.correct / (metrics.call_n + matrix.sum(axis=0).to_numpy())
    return metrics, matrix, calls.fold.nunique()


def propagation():
    # File blocks 03–05, 06–08 and 09–11 each follow flat, mod, tv order.
    frames = []
    for i, measure in enumerate(MEASURES):
        for j, group in enumerate(GROUPS):
            wide = raw_csv(3 + 3 * i + j)
            long = wide.melt(var_name="distance", value_name="correlation")
            long["distance_m"] = long.distance.str.removesuffix("m").astype(int)
            frames.append(long.assign(measure=measure, call_type=group))
    rows = pd.concat(frames, ignore_index=True)
    by = ["measure", "call_type", "distance_m"]
    summary = propagation_summary(rows)
    spread = rows.groupby(by).correlation.quantile([.25, .75]).unstack()
    spread.columns = ["q25", "q75"]
    summary = summary.join(spread, on=by)
    reference = pd.read_csv(ROOT / "04_function_and_transmission/tables/propagation_by_measure_distance.csv")
    pd.testing.assert_frame_equal(summary[reference.columns].sort_values(by).reset_index(drop=True),
                                  reference.sort_values(by).reset_index(drop=True))
    return summary


def playback_rates(rows):
    records = []
    for label in (*GROUPS, "control", "all_calls"):
        group = rows[rows.Type.eq("call")] if label == "all_calls" else rows[rows.CallType.eq(label)]
        for outcome in OUTCOMES:
            n, count = len(group), int(group[outcome].sum())
            ci = stats.binomtest(count, n).proportion_ci(method="wilson")
            records.append((label, outcome, n, count, count / n, ci.low, ci.high))
    return pd.DataFrame(records, columns=["type", "outcome", "n", "positive", "rate", "low", "high"])


def clustered_logit(design, outcome, clusters, terms):
    """Unpenalized binomial MLE; cluster sandwich CR1, t intervals and joint Wald F."""
    x, y = design.to_numpy(dtype=float), outcome.to_numpy(dtype=float)
    n, k = x.shape
    def loss(beta):
        eta = x @ beta
        return np.sum(np.logaddexp(0, eta) - y * eta)

    def gradient(beta):
        return x.T @ (special.expit(x @ beta) - y)

    fit = optimize.minimize(loss, np.zeros(k), jac=gradient, method="BFGS", options={"gtol": 1e-8})
    if not np.isfinite(fit.x).all() or np.max(np.abs(gradient(fit.x))) > 1e-5:
        raise RuntimeError(f"Logistic fit did not converge for {outcome.name}: {fit.message}")
    prob = special.expit(x @ fit.x)
    bread = np.linalg.inv((x.T * (prob * (1 - prob))) @ x)
    scores = pd.DataFrame(x * (y - prob)[:, None]).groupby(clusters.to_numpy()).sum().to_numpy()
    g = len(scores)
    cov = (g / (g - 1) * (n - 1) / (n - k)) * bread @ (scores.T @ scores) @ bread
    ix = design.columns.get_indexer(terms)
    beta, subcov = fit.x[ix], cov[np.ix_(ix, ix)]
    q = len(terms)
    wald = (beta @ np.linalg.solve(subcov, beta)) / q
    joint = dict(n=n, clusters=g, df1=q, df2=g-1, F=wald, p=stats.f.sf(wald, q, g-1))
    radius = stats.t.ppf(.975, g-1) * np.sqrt(np.diag(subcov))
    effects = pd.DataFrame({"term": terms, "OR": np.exp(beta), "low": np.exp(beta-radius),
                            "high": np.exp(beta+radius)})
    return joint, effects


def adjusted_playback(rows):
    tests, estimates = [], []
    for sample in ("natural_types", "call_vs_control"):
        natural_only = sample == "natural_types"
        d = rows[rows.Type.eq("call")] if natural_only else rows
        design = pd.DataFrame({"intercept": 1., "searching": d.Context.eq("searching"),
                               "after": d.Context.eq("after"), "log_distance": np.log(d.Distance),
                               "log1p_group": np.log1p(d.Npenguins)})
        if natural_only:
            terms = ["mod_vs_flat", "tv_vs_flat"]
            design[terms[0]], design[terms[1]] = d.CallType.eq("mod"), d.CallType.eq("tv")
            outcomes, cluster = OUTCOMES, "Date"
        else:
            terms = ["call_vs_control"]
            design[terms[0]] = d.Type.eq("call")
            outcomes, cluster = ["ResponseYesNo"], "expID"
        for outcome in outcomes:
            joint, effects = clustered_logit(design, d[outcome], d[cluster], terms)
            tests.append(dict(sample=sample, outcome=outcome, cluster=cluster, **joint))
            estimates.append(effects.assign(sample=sample, outcome=outcome, cluster=cluster))
    tests = pd.DataFrame(tests)
    mask = tests["sample"].eq("natural_types")
    tests.loc[mask, "p_holm_four"] = holm(tests.loc[mask, "p"])
    return tests, pd.concat(estimates, ignore_index=True)


def first_reply(calls):
    vr = [f"VR{i}" for i in range(1, 20)]
    mismatches = calls.loc[calls.VocalYesNo.eq(0) & calls[vr].notna().any(axis=1), ["expID", "CallType", *vr]]
    voiced = calls[calls.VocalYesNo.eq(1)]
    table = pd.crosstab(voiced.CallType, voiced.VR1).reindex(index=GROUPS, fill_value=0)
    table = table.reindex(columns=[*GROUPS, *sorted(set(table.columns) - set(GROUPS))], fill_value=0)
    obs = table.to_numpy()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / obs.sum()
    stat = np.sum((obs - expected)**2 / expected)
    rng, exceed = np.random.default_rng(SEED), 0
    for start in range(0, SIMULATIONS, 10000):
        sampled = stats.random_table.rvs(obs.sum(axis=1), obs.sum(axis=0),
                                         size=min(10000, SIMULATIONS-start), random_state=rng)
        values = np.sum((sampled - expected)**2 / expected, axis=(1, 2))
        exceed += np.count_nonzero(values >= stat - 1e-12)
    test = dict(n=int(obs.sum()), missing_VR1=int(voiced.VR1.isna().sum()), chi2=stat,
                p=(exceed+1)/(SIMULATIONS+1), draws=SIMULATIONS, seed=SEED)
    return table, test, mismatches


def draw_playback(rates, reply):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=(2, 1),
                                  layout="constrained")
    for shift, group in zip((-.25, 0, .25), GROUPS):
        t = rates[rates.type.eq(group)].set_index("outcome").loc[list(OUTCOMES)]
        x = np.arange(len(t)) + shift
        ax.bar(x, t.rate, width=.22, color=COLORS[group], label=f"{group} (n={t.n.iloc[0]})")
        ax.errorbar(x, t.rate, yerr=[t.rate-t.low, t.high-t.rate], fmt="none", color=".18", capsize=3)
    ax.set(xticks=np.arange(4), xticklabels=["Any", "Heads up", "Vocal", "Approach"],
           ylabel="Proportion of playbacks", title="Responses to natural calls")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    for j, group in enumerate(("all_calls", "control")):
        r = rates[(rates.type == group) & (rates.outcome == "ResponseYesNo")].iloc[0]
        bx.bar(j, r.rate, width=.55, color=["#446987", COLORS["control"]][j])
        bx.errorbar(j, r.rate, yerr=[[r.rate-r.low], [r.high-r.rate]], color=".18", capsize=4)
        bx.text(j, r.high+.025, f"{r.positive}/{r.n}", ha="center")
    bx.set(xticks=[0, 1], xticklabels=["Natural", "Control"], title="Any response in paired trials")
    for axis in (ax, bx):
        axis.set_ylim(0, 1)
        axis.spines[["right", "top"]].set_visible(False)
    fig.savefig(OUT / "playback_rates.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.7, 3.8), layout="constrained")
    ax.imshow(reply, cmap="Blues", vmin=0)
    for (i, j), count in np.ndenumerate(reply.to_numpy()):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > reply.to_numpy().max() * .6 else "black")
    ax.set(xticks=range(len(reply.columns)), xticklabels=reply.columns, xlabel="First vocal reply",
           yticks=range(len(reply)), yticklabels=reply.index, ylabel="Playback call type",
           title=f"First reply per vocal trial (n={reply.to_numpy().sum()})")
    fig.savefig(OUT / "first_reply.png", dpi=180)
    plt.close(fig)


def evidence_by_type(metrics, prop, rates, effects):
    out = metrics.copy()
    for distance in (128, 256):
        part = prop[prop.distance_m.eq(distance)].pivot(index="call_type", columns="measure", values="mean")
        out = out.join(part.add_suffix(f"_{distance}m"))
    for outcome in OUTCOMES:
        part = rates[rates.outcome.eq(outcome)].set_index("type")
        out[f"{outcome}_positive"] = part.positive
        out[f"{outcome}_n"] = part.n
        out[f"{outcome}_rate"] = part.rate
    # Include adjusted type contrasts beside the raw rates. Flat is the reference.
    part = effects[(effects["sample"] == "natural_types") & (effects.outcome == "ApproachYesNo")].copy()
    part.index = part.term.str.removesuffix("_vs_flat")
    out = out.join(part[["OR", "low", "high"]].add_prefix("approach_vs_flat_"))
    out.loc["flat", "approach_vs_flat_OR"] = 1.
    return out


def choose_measures():
    ptable = pd.read_csv(ROOT / "03_nonparametric/tables/univariate_tests.csv").set_index("feature")
    rows = [
        ("信号衰减", "三种传播相关系数", "现有响应量", "时频图相关衡量整体结构，幅度谱与包络相关分别衡量频谱和时间结构保留。以类型、距离及交互解释。"),
        ("信号衰减", "Fpeak（主）、Q50（辅）", "逐录音提取后可用",
         f"Fpeak 在 Data01 的类间差异经 Holm 校正 p={ptable.loc['Fpeak','p_holm7']:.4f}。Q50 补充能量中心。需匹配参考录音与各距离录音后计算变化。"),
        ("信号衰减", "PulseRate、F02−F01", "条件性候选", "分别检验时间调制及双声结构随距离的保持；仅对能测脉冲或双基频的录音计算。"),
        ("行为反应", "播放刺激类型 CallType", "现有声学解释变量", "可直接对应每次播放。分别分析总反应、抬头、鸣叫、靠近，控制情境、距离与群体大小。"),
        ("行为反应", "Fpeak、PulseRate、双基频间距", "需匹配 PlaybackTrack", "主导频率、脉冲节律、双声结构是可检验的候选线索。先提取每条播放轨的实测值，再估计逐轨行为关联。"),
        ("两类模型", "Duration、F01、Q25/Q75", "备选描述量", "描述时长、基频和频谱两端。直接估计连续特征效应需要对应音轨，纳入多个频率量前需检查共线性。"),
    ]
    return pd.DataFrame(rows, columns=["模型", "指标", "使用条件", "选择理由"])


def md_table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(x) for x in cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def write_report(evidence, matrix, folds, tests, effects, rates, reply_test, anomalies,
                 selection, different_distance):
    n = int(evidence.call_n.sum())
    hits = int(evidence.correct.sum())
    raw_rates = rates[rates.outcome.eq("ResponseYesNo")].set_index("type")
    def count_rate(group):
        r = raw_rates.loc[group]
        return f"{int(r.positive)}/{int(r.n)}（{r.rate:.1%}）"
    precision_rows, acoustic_rows, prop_rows = [], [], []
    for group, r in evidence.iterrows():
        mistakes = "、".join(f"{other} {matrix.loc[group, other]} 条" for other in GROUPS
                              if other != group and matrix.loc[group, other])
        precision_rows.append((group, f"{int(r.correct)}/{int(r.call_n)}", f"{r.precision:.1%}",
                               f"{r.recall:.1%}", f"{r.F1:.3f}", mistakes or "无"))
        acoustic_rows.append((group, f"{int(r.correct)}/{int(r.call_n)}", f"{int(r.F02_measured)}/{int(r.call_n)}",
                              f"{int(r.pulse_measured)}/{int(r.call_n)}", f"{r.Fpeak_median_Hz:g}",
                              f"{r.spectrogram_128m:.3f}", count_rate(group),
                              f"{int(r.ApproachYesNo_positive)}/{int(r.ApproachYesNo_n)}（{r.ApproachYesNo_rate:.1%}）"))
        for dist in (128, 256):
            prop_rows.append((group, dist, *(f"{r[f'{m}_{dist}m']:.3f}" for m in MEASURES)))
    type_tests = tests[tests["sample"].eq("natural_types")].set_index("outcome")
    natural = effects[effects["sample"].eq("natural_types")].set_index(["outcome", "term"])
    def interval(r):
        return f"{r.OR:.2f}［{r.low:.2f}，{r.high:.2f}］"
    behavior_rows = [(label, interval(natural.loc[(key, "mod_vs_flat")]),
                      interval(natural.loc[(key, "tv_vs_flat")]), f"{type_tests.loc[key,'p']:.4f}",
                      f"{type_tests.loc[key,'p_holm_four']:.4f}") for key, label in OUTCOMES.items()]
    found = [OUTCOMES[key] for key, r in type_tests.iterrows() if r.p_holm_four < .05]
    conclusion = ("通过四结局 Holm 校正的项目为" + "、".join(found) if found else
                  "四项结局的类型整体检验经 Holm 校正均未达到 0.05，当前数据不足以确认三类自然声的行为概率有差别")
    control = effects[effects["sample"].eq("call_vs_control")].iloc[0]
    pair_test = tests[tests["sample"].eq("call_vs_control")].iloc[0]
    vr_conclusion = ("检出关联" if reply_test["p"] < .05 else "未检出关联")
    parts = [
        "# 第一题第四小问 · 声学判别、传播保真与海上回放",
        "以真实叫声类型为对应单位。Data01 提供已标注鸣声的折外判别结果，Data03–11 提供随距离变化的相关系数，Data02 提供真实播放刺激后的行为。三批观测缺少可核验的逐录音映射。",
        "## 三类叫声的综合证据",
        md_table(["叫声", "折外判对", "F02 有测值", "脉冲率有测值", "Fpeak 中位数（Hz）", "128 m 时频图相关", "总体响应", "靠近"], acoustic_rows),
        "**flat** 的样本峰值频率中位数最高，128 m 时结构保留较弱。播放后常见鸣叫，靠近较少，符合维持联系的解释。"
        "\n\n**mod** 有可测脉冲的样本较多，256 m 的包络平均相关最高。靠近比例在样本中最高，使时间调制成为后续检验定位或招募作用的候选线索。"
        "\n\n**tv** 的双基频可帮助判别，128 m 三种结构的平均相关最高，256 m 时仍以时频与幅度谱保留见长。回放可引出鸣叫和靠近；远距离结构保留与具体行为功能应分别解读。"
        "\n\n这些观察支持联系功能的假设。整段声音播放同时包含多种声学成分，现有设计无法单独辨认峰值频率、拍频或节律产生的行为效应。",
        "## 分类结果与录音批次",
        f"读取已有三层分类器的 blank6_stack 折外结果，按日期与地点留出 {folds} 批。整体准确率 {hits}/{n}（{hits/n:.2%}），Macro-F1 为 {evidence.F1.mean():.4f}。前两层使用字段是否有测值，结果依赖原有测量流程。真实声学量由 Data01 原始 CSV 按源行号、日期、地点及类别核对。",
        md_table(["真实类型", "正确数", "精确率", "召回率", "F1", "误判去向"], precision_rows),
        "## 传输稳健性与距离",
        md_table(["叫声", "距离（m）", "时频图相关", "幅度谱相关", "包络相关"], prop_rows),
        "以上为相对 1 m 参照的有效相关系数均值。完整表列出 4–256 m 七个距离的总数、有效数、均值、标准差、中位数及四分位数。同一录音可参与多次相关计算，四分位数描述观测分布。结构相似度与企鹅的实际听觉识别距离属于不同测量。",
        "## 回放行为与功能解释",
        f"自然叫声的总体响应为 {count_rate('all_calls')}，对照声为 {count_rate('control')}。共 {int(pair_test.clusters)} 对实验，其中 {different_distance} 对的两次播放距离不同。以总体响应为结局，调整协变量后，自然声相对对照声的优势比 OR 为 {interval(control)}，p={pair_test.p:.3g}。方括号内为 95% 区间。",
        md_table(["自然声结局", "mod 对 flat 的 OR［95% 区间］", "tv 对 flat 的 OR［95% 区间］", "类型整体 p", "四项 Holm p"], behavior_rows),
        f"{conclusion}。靠近的组间点估计值得跟进，区间较宽。上表的单个对比区间未经多重校正，关于三类功能的整体判断采用最后一列。",
        f"自然声模型为四个二项 logistic 回归。每个结局的线性预测项均为 CallType + Context + log(Distance) + log(1+Npenguins)，参照为 flat 和 commuting。按 {int(type_tests.clusters.iloc[0])} 个日期聚类，采用 CR1 协方差、t 区间和联合 Wald F 检验；日期簇较少，推断为探索性。自然声与对照声的总体响应模型将 CallType 换为 Type，按 expID 聚类，控制相同协变量，对照声为参照。原始比例图显示边际 Wilson 95% 区间。重复使用的播放轨限制了对新音轨的推广。",
        f"首次鸣叫回应分析仅纳入自然声试次中 VocalYesNo=1 且 VR1 有值的 {reply_test['n']} 次，每次取第一声。另有 {reply_test['missing_VR1']} 次已记录鸣叫但 VR1 缺失。固定边际 Pearson 统计量 {reply_test['chi2']:.3f}，Monte Carlo p={reply_test['p']:.4f}（{SIMULATIONS:,} 次，种子 {SEED}），播放类型与首次回应类型{vr_conclusion}。自然声中另有 {len(anomalies)} 条 VocalYesNo=0 却出现 VR 记录，详见异常表。",
        "## 后续模型的指标",
        md_table(selection.columns, selection.itertuples(index=False, name=None)),
        "## 复核与复现",
        "在 VS Code 中打开合并包，选择装有 NumPy、pandas、SciPy 与 Matplotlib 的 Python 解释器，运行 [problem1_integration.py](../../APMCM_C_Q1_Full/code/problem1_integration.py)。命令行也可从合并包根目录执行 `python APMCM_C_Q1_Full/code/problem1_integration.py`。结果写入本报告所在目录。",
        "[逐类综合表](evidence_by_type.csv) · [混淆矩阵](confusion.csv) · [各距离传播统计](propagation.csv) · [三通道传播图](../../APMCM_C_Q1_Full/04_function_and_transmission/figures/propagation_profiles.png) · [回放比例与区间](playback_rates.csv) · [回放图](playback_rates.png) · [调整后整体检验](adjusted_tests.csv) · [调整后优势比](adjusted_effects.csv) · [首次回应矩阵](VR1_counts.csv) · [首次回应热图](first_reply.png) · [首次回应检验](VR1_test.csv) · [自然声 VR 异常](VR_flags.csv) · [指标清单](selected_measures.csv)",
    ]
    (OUT / "第四小问结果.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main():
    metrics, matrix, folds = classification()
    prop = propagation()
    rows = raw_csv(2)
    fields = [*OUTCOMES, "CallType", "Type", "Context", "Date", "expID", "Distance", "Npenguins"]
    if (rows[fields].isna().any().any() or not rows[list(OUTCOMES)].isin([0, 1]).all().all()
            or not rows.Context.isin(["commuting", "searching", "after"]).all()
            or not rows.Distance.gt(0).all() or not rows.Npenguins.ge(0).all()):
        raise ValueError("Check missing or invalid model inputs in Data02.")
    calls = rows[rows.Type.eq("call")]
    pair_counts = pd.crosstab(rows.expID, rows.Type)
    if (set(pair_counts.columns) != {"call", "control"} or not pair_counts.eq(1).all().all()
            or not calls.CallType.isin(GROUPS).all() or not rows.loc[rows.Type.eq("control"), "CallType"].eq("control").all()):
        raise ValueError("Check pair membership and stimulus labels in Data02.")
    different_distance = int(rows.groupby("expID").Distance.nunique().gt(1).sum())
    rates = playback_rates(rows)
    tests, effects = adjusted_playback(rows)
    reply, reply_test, anomalies = first_reply(calls)
    selection = choose_measures()
    evidence = evidence_by_type(metrics, prop, rates, effects)
    OUT.mkdir(parents=True, exist_ok=True)
    for table, name in [(evidence.reset_index(), "evidence_by_type"), (matrix.reset_index(), "confusion"),
                        (prop, "propagation"), (rates, "playback_rates"), (tests, "adjusted_tests"),
                        (effects, "adjusted_effects"), (reply.reset_index(), "VR1_counts"),
                        (pd.DataFrame([reply_test]), "VR1_test"), (anomalies, "VR_flags"),
                        (selection, "selected_measures")]:
        save(table, name)
    draw_playback(rates, reply)
    write_report(evidence, matrix, folds, tests, effects, rates, reply_test, anomalies,
                 selection, different_distance)
    print(OUT / "第四小问结果.md")


if __name__ == "__main__":
    main()
