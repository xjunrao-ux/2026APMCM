"""E. Compare independent sources at call-type level and write the first-question findings."""
import pandas as pd

from acoustics import LABELS, OUT, report, save, source, table


def experiments():
    pb = pd.read_csv(source(2))
    events = ["ResponseYesNo", "HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]
    if not pb[events].isin([0, 1]).all().all():
        raise ValueError("Playback responses require review before computing rates.")
    playback = pb.groupby("CallType")[events].sum()
    playback.insert(0, "n", pb.groupby("CallType").size())
    for c in events:
        playback[c+"_rate"] = playback[c]/playback.n
    save(playback.reset_index(), "playback_by_type.csv")
    rows = []
    for i in range(3, 12):
        frame = pd.read_csv(source(i))
        metric = ["spectrogram", "spectrum", "envelope"][(i-3)//3]
        call = LABELS[(i-3)%3]
        for dist, values in frame.items():
            x = values.dropna()
            if not x.between(-1, 1).all():
                raise ValueError(f"{source(i)} contains invalid correlations")
            rows.append([i, call, metric, int(dist.removesuffix("m")), len(x), x.mean(), x.std()])
    prop = save(pd.DataFrame(rows, columns=["source_no", "CallType", "metric", "distance_m", "n", "mean_r", "sd"]),
                "propagation_by_distance.csv")
    return playback, prop, pb.expID.nunique()


def conclude(desc, tests, models, classes, coef):
    playback, prop, n_exp = experiments()
    kw = tests.query("test == 'Kruskal-Wallis'")[["feature", "statistic", "p_holm", "effect"]].copy()
    blocked = pd.read_csv(OUT / "within_date_permutation.csv")
    kw["p_date_holm"] = kw.feature.map(blocked.set_index("feature").p_holm)
    peaks = desc[desc.feature.eq("Fpeak")].set_index("CallType")
    peakmw = tests.query("feature == 'Fpeak' and groups == 'flat/mod'").iloc[0]
    pulse = tests[tests.feature.eq("PulseRate")].iloc[0]
    bytype = classes[classes.scheme.eq("date")]
    tv = bytype[bytype.CallType.eq("tv")].set_index("model")
    tv_n = int(tv.loc["L2", "n"])
    adj = pd.read_csv(OUT / "weight_sensitivity.csv").iloc[0]
    importance = pd.read_csv(OUT / "rf_permutation.csv").groupby("feature").mean_macro_f1_drop.mean().sort_values(ascending=False)
    odds = coef.pivot(index="feature", columns="contrast", values="odds_ratio")
    remote = prop[prop.distance_m.isin([128, 256])].pivot(index=["distance_m", "CallType"], columns="metric", values="mean_r").reset_index()
    natural = playback.loc[LABELS, ["n", "ResponseYesNo"]].sum()
    report("problem1_conclusion.md", f"""# 第一问结论 · 非洲企鹅海面叫声

## 三类叫声中最清楚的差别

flat 的峰值频率中位数为 {peaks.at['flat','median']:g} Hz，mod 为 {peaks.at['mod','median']:g} Hz，tv 为 {peaks.at['tv','median']:g} Hz。共性特征中，峰值频率的总体秩差异最大。flat 对 mod 的 Cliff’s δ = {peakmw.effect:.3f}，18 项两两比较统一 Holm 校正后 p = {peakmw.p_holm:.4f}。其余两对的峰频比较未达校正阈值。

{table(kw)}

statistic 为 Kruskal–Wallis H，effect 为秩 epsilon²。常规总体检验将每条叫声视作独立观测；日期内置换得到的 p_date_holm 反映控制各日类别组成后的敏感性结果。F01 在两种分析中的结论随条件改变，可作为候选次要指标。Duration 和频谱分位数均有较大的类别重叠。F02 只在 tv 有记录，描述双声结构。mod 与 tv 的实测脉冲率没有显示明确的秩差异（U = {pulse.statistic:g}，δ = {pulse.effect:.3f}，p = {pulse.p_holm:.3f}）；存在漏测，不能断言两类等同。

## 分类效力与声学解释

以下为全部样本折外预测汇总，date 为跨录音日主评估，stratified 为普通分层对照。

{table(models[["scheme","model","accuracy","macro_f1","balanced_accuracy"]])}

随机森林在日期验证下的 Macro-F1 最高，但仅识别出 {int(round(tv.loc['RF','recall']*tv_n))}/{tv_n} 条 tv；L2 仅识别出 {int(round(tv.loc['L2','recall']*tv_n))}/{tv_n} 条。固定把全部录音判为 flat 可得到 44/108 = 40.7% 的准确率。六项汇总特征对新日期录音的三分类能力有限。模型排名仅反映这 16 个日期；对预测模型的事后比较还需独立录音批次验证。

{table(bytype[["model","CallType","precision","recall","f1","n"]])}

针对 tv 漏判，对 L2 在相同日期折中按训练类别频数反比加权并重新调 C。折外 tv 召回为 {adj.tv_recall:.3f}，Macro-F1 为 {adj.macro_f1:.3f}，准确率为 {adj.accuracy:.3f}。该补充比较由原模型表现触发，须另取数据确认。

全量拟合的标准化 L2 模型中，其他输入固定时，Fpeak 每增加一个样本总体标准差（StandardScaler，ddof=0），mod 对 flat 的相对优势乘以 {odds.at['Fpeak','mod/flat']:.3f}，tv 对 flat 乘以 {odds.at['Fpeak','tv/flat']:.3f}。F01 对 tv/flat 的相应倍数为 {odds.at['F01','tv/flat']:.3f}。这些是收缩后的条件关联估计。森林的外折单特征置换中，平均 Macro-F1 降幅最大的是 {importance.index[0]}（{importance.iloc[0]:.3f}）。该指标衡量特定模型预测对输入的依赖，相关变量可替代彼此，小测试折下排名会波动。

## 传播与播放证据如何衔接

以下是以 1 m 信号为参考的平均相关系数，各列分别为包络、声谱图和幅度谱相关。每个距离与指标的有效数、标准差见 propagation_by_distance.csv。

{table(remote)}

128 m 处 tv 三项平均相关均最高。到 256 m，tv 的声谱图和幅度谱仍占优，包络相关则以 mod 最高。保真度的排序取决于距离与表征；特别是 flat 在录音中较高的峰值频率，并未对应最高的远距离平均频谱相关。

播放数据含 {n_exp} 个实验编号。各行为分母均为该播放类型的全部试次，行为可同时发生。自然叫声总体响应为 {natural.ResponseYesNo}/{natural.n}（{natural.ResponseYesNo/natural.n:.1%}），对照为 {playback.at['control','ResponseYesNo']}/{playback.at['control','n']}（{playback.at['control','ResponseYesNo_rate']:.1%}）。

{table(playback.reset_index()[["CallType","n","ResponseYesNo_rate","HeadsUpYesNo_rate","VocalYesNo_rate","ApproachYesNo_rate"]])}

flat 的抬头反应与注意定向相符，mod、tv 的接近比例均高于 flat；tv 的总体响应比例最高。结合三类录音分类重叠、tv 在较远距离保留频谱结构的现象，可提出三类叫声都参与海上联系、部分声学线索可能与接近或回应有关的假设。原始比例受情境、距离、群体大小和实验配对影响，无法由此确认叫声表达的具体信息，也无法将某个频率值与单次行为对应。

第二问优先比较频谱保真度，保留包络维度，并用 Fpeak、F01 和频谱分位数描述可能损失的线索；若要估计某一频率线索自身的衰减，需补各距离的对应声学测量。第三问使用 Data02 的真实 CallType、Context、Distance、Npenguins 与 expID，按真实类别解释反应；F02 和 PulseRate 可用于设计后续刺激。第四问可在类别和距离层面结合保真度与情境调整后的响应概率。三套实验未提供可验证的逐条对应键。
""")
