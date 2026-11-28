from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from models import BEHAVIORS, CALLS, COMBOS, CONTEXTS

OUT=Path(__file__).resolve().parent/"results"


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                       *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def report(d,avg,val,times,scores,cont,st,effects,structure,family,n_boot,plus,minus,
           paired_p,combo_p,score_p,glmm_sd,glmm,intensity,context_effects,curves,output=OUT):
    def interval(r,key,k=1):
        return f"{k*r[key]:.2f} [{k*r[key+'_lo']:.2f}, {k*r[key+'_hi']:.2f}]"
    natural=cont[cont["first"]=="natural_equal"].iloc[0]
    rates=d.groupby("Type").ResponseYesNo.agg(["sum","size","mean"])
    pairs=d.expID.nunique()
    size=d.Npenguins.median()
    context_counts=d.drop_duplicates("expID").Context.value_counts()
    weights="、".join(f"{name} {context_counts[c]}/{pairs}" for c,name in
                     zip(CONTEXTS,["进食后","赶路","搜寻"]))
    against_control=cont[(cont["second"]=="control") & cont["first"].isin(CALLS)]
    example=avg.query("CallType=='flat' and Distance==100 and behavior=='HeadsUp'").iloc[0]
    example_curve=curves.query("CallType=='flat' and Distance==100 and Context=='standardized' and behavior=='HeadsUp'")
    diagnostics=pd.read_csv(output/"r_baseline_diagnostics.csv")
    coefficients=pd.read_csv(output/"q3_supplementary3_parameters.csv")
    baseline=pd.read_csv(output/"q3_code1_comparison.csv")
    acoustic=pd.read_csv(output/"q3_supplementary4.csv")
    selected_terms=[("HeadsUp","hu_CallTypeflat"),("HeadsUp","scaleNpenguins"),
                    ("Vocal","VocalNcallsBefore"),("Approach","CallTypetv")]
    selected_coef=coefficients.set_index(["behavior","term"]).loc[selected_terms].reset_index()
    def interval_direction(low, high):
        return np.select([low>0,high<0], [1,-1], default=0)
    published_direction=interval_direction(coefficients.published_lo95,coefficients.published_hi95)
    refit_direction=interval_direction(coefficients.lo95,coefficients.hi95)
    different=coefficients.loc[published_direction!=refit_direction]
    agreement=f"按 95% 区间完全高于零、完全低于零或包含零核对，{len(coefficients)-len(different)}/{len(coefficients)} 项分类一致。"
    if not different.empty:
        agreement+="分类有差异的参数为 "+ "、".join(f"{r.behavior} 的 {r.term}" for r in different.itertuples())+"。"
    agreement+="原表保留两位小数，接近零的区间端点对舍入和采样波动较敏感。"
    res=["# 第三问运行结果",
         f"{pairs} 个配对实验中，自然叫声后有 {rates.loc['call','sum']}/{pairs} 组出现响应（{100*rates.loc['call','mean']:.1f}%），控制声后有 {rates.loc['control','sum']}/{pairs} 组（{100*rates.loc['control','mean']:.1f}%）。只对自然声响应的有 {plus} 对，只对控制声响应的有 {minus} 对。双侧精确 McNemar 检验 p = {paired_p:.3g}。",
         f"在相同的情境、距离和群体规模分布上，将三种自然声等权平均，相对控制声的调整概率差为 {interval(natural,'risk_difference',100)} 个百分点，边际优势比为 {interval(natural,'marginal_OR')}。第三问主模型使用按实验配对重抽样所得的点态 95% 区间。",
         "## 哪些条件与响应有关",
         table(["声型","实测响应数／实验数","共同协变量分布下的调整概率，%"],
               [[r.CallType,f"{d.loc[d.CallType==r.CallType,'ResponseYesNo'].sum()}/{sum(d.CallType==r.CallType)}",
                 interval(r,"p_response",100)] for _,r in st.iterrows()]),
         table(["对比","调整概率差，百分点","边际优势比"],
               [[r["first"]+" 相对 "+r["second"],interval(r,"risk_difference",100),interval(r,"marginal_OR")]
                for _,r in cont.iterrows() if r["first"]!="natural_equal"]),
         table(["情境","自然声响应数／实验数","控制声响应数／实验数"],
               [[c,*[f"{g.ResponseYesNo.sum()}/{len(g)}" for typ in ["call","control"]
                      for g in [d[(d.Context==c)&(d.Type==typ)]]]] for c in CONTEXTS]),
         "以下三种情境均采用全部回放的相同距离与群体规模分布，三种自然声等权。每种声型的分情境对比见 [情境对比表](q3_context_contrasts.csv)。",
         table(["情境","调整自然声概率，%","调整控制声概率，%","调整差，百分点（95% 区间）"],
               [[r.Context,f"{100*r.p_call:.2f}",f"{100*r.p_control:.2f}",interval(r,"risk_difference",100)]
                for _,r in context_effects[context_effects.CallType=="natural_equal"].iterrows()]),
         f"叫声与情境六项交互系数的联合得分检验 p = {score_p:.3g}。全数据按组五折选中的结构为 {structure}。距离增加 10 m 的平均预测概率变化为 {interval(effects.iloc[0],'risk_difference',100)} 个百分点，群体增加 1 只为 {interval(effects.iloc[1],'risk_difference',100)} 个百分点。计算仅使用变化后仍处于总体观测范围的行。",
         "## 一次回放后会出现哪些行为组合",
         f"H 为抬头、V 为鸣叫、A 为接近。{pairs} 对实验内交换一次回放的完整组合标签，共进行 50,000 次置换。八类边际分布相同的全局检验 p = {combo_p:.5f}（蒙特卡洛下限 1/50,001）。",
         table(["组合","自然声次数","控制声次数","等权自然声相对控制声的调整概率差，百分点"],
               [[c,sum((d.Type=="call")&(d.code==j)),sum((d.Type=="control")&(d.code==j)),
                 interval(natural,"difference_"+c,100)] for j,c in enumerate(COMBOS)]),
         table(["声型相对控制声","仅抬头变化，百分点","抬头＋鸣叫变化，百分点"],
               [[r["first"],interval(r,"difference_H",100),interval(r,"difference_HV",100)]
                for _,r in against_control.iterrows()]),
         "三种自然声各自对控制声的全部八类调整差值见 [组合对比表](q3_contrasts.csv)。稀有组合的数值受到收缩强度影响，逐项区间未做多重比较校正。配对置换依赖配对内可交换假设，调整模型使用每次回放的实际距离。",
         f"## 群体规模为 {size:g} 只时，通常多久出现响应",
         f"预测单元依次列出「60 秒内发生概率（%）／发生者首次响应中位秒数」。情境权重为{weights}。各情境的时间分布按事件概率加权混合后求中位数。接近时间合并 {int(d.ApproachYesNo.sum())} 次事件估计。最右列给出完全匹配声型、距离、群大小的实际回放数，以及其中抬头、鸣叫、接近的事件数。"]
    rows=[]
    for (c,dist),grp in avg.groupby(["CallType","Distance"],sort=False):
        if c=="control":
            continue
        g=grp.set_index("behavior")
        n=int(g.n_exact.iloc[0])
        status=(f"{n} 次；"+"/".join(str(int(g.loc[h,"events_exact"])) for h in BEHAVIORS)) if n else "无该条件观测"
        cells=[]
        for h in BEHAVIORS:
            estimate=f"{100*g.loc[h,'p_event_60']:.1f}／{g.loc[h,'median_if_event_s']:.1f}"
            cells.append("实测无响应（模型 "+estimate+"）" if n and not g.loc[h,"events_exact"] else estimate)
        rows.append([c,dist,*cells,status])
    res.extend([table(["叫声","距离 m","抬头 %／秒","鸣叫 %／秒","接近 %／秒","实测回放；H/V/A 事件数"],rows),
                "在有观测的条件下，事件数为 0 表示该行为未响应。完整[潜伏期表](q3_latency_predictions.csv)另标出距离超出对应声型和情境实测范围的预测。对某声型从未观察到的行为，发生者中位时间留空。未观察到行为不能判断企鹅是否听到了声音。",
                f"以 flat、100 m、{size:g} 只的抬头为例，60 秒发生概率为 {100*example.p_event_60:.2f}%。发生者的一半对应全部回放的 {50*example.p_event_60:.2f}%，累计概率到此水平的时间约为 {example.median_if_event_s:.2f} 秒。该条件实测 {example.n_exact} 次，抬头 {example.events_exact} 次。",
                table(["播放后，秒","全部回放累计发生，%","窗内响应者累计发生，%"],
                      [[f"{r.time_s:g}",f"{100*r.p_event_by_t:.2f}",f"{100*r.p_by_t_given_event:.2f}"]
                       for r in example_curve[example_curve.time_s.isin([1,1.5,1.75,2,60])].itertuples()]),
                "所有代表条件的分时点概率及区间见[累计概率曲线](q3_latency_curves.csv)。[逐次回放表](q3_trial_responses.csv)记录实际最早事件和秒数；同秒开始的行为全部保留。",
                "## 响应强度",
                f"下表选取 searching、100 m、{size:g} 只、播放前 60 秒鸣叫数为样本中位数的情境。每格依次为「发生正值的概率（%）／正值条件均值／每次回放的期望」。",
                table(["叫声","抬头（%／秒／秒）","鸣叫（%／次／次）","接近（%／秒／秒）"],
                      [[c,*[f"{100*r.p_positive:.1f}／{r.mean_if_positive:.1f}／{r.expected_all:.2f}"
                             for _,r in intensity[(intensity.CallType==c)&(intensity.Distance==100)&
                                                 (intensity.Context=="searching")].iterrows()]] for c in CALLS]),
                f"接近正持续时间有 {d.ApproachDuration.gt(0).sum()} 条，表中正值时长使用合并均值。原始鸣叫次数包含播放前已有鸣叫活动的群体，故其正值概率与鸣叫响应概率可不同。",
                "## 复现与第四问接口",
                f"运行 `python run_q3.py` 会调用 R 执行 Code1，然后生成第三问预测和两份附件对照。本次按 expID 重抽样 {n_boot} 次，Python 随机种子 20261127。环境依赖和命令见[程序运行思路](../第三题的程序运行思路.md)。",
                "第四问读取 q3_behavior_combo_predictions.csv 的 p_all 即可取得一次群体回放的八类概率。q3_latency_predictions.csv 的 Context=standardized 为情境综合行，其余为分情境行。q3_overall_response_predictions.csv 与 q3_intensity_predictions.csv 给出总体响应概率和三项强度。列名后缀 _lo、_hi 对应点态 95% 区间。频数、置信区间和群体规模均以群体回放为解释单位。"])
    ref=diagnostics.iloc[0]
    subset=baseline.query("Context=='searching' and Distance==100 and CallType=='flat'")
    flat=acoustic.loc[acoustic.CallType.eq("flat")].sort_values("Distance")
    res.extend(["## 附件基准在第三问中回答什么",
        f"Code1 直接读取 Data02，生成行为组合和自然声／控制声频数。脚本沿用作者选定的三个 brms 模型、先验及随机效应。本次各模型运行 {int(ref.chains)} 条链，每链 {int(ref.iterations)} 次迭代，含 {int(ref.warmup)} 次预热。补充三使用同一批实验。下表逐项核对参数的后验均值和 95% 可信区间。[完整系数与 PDF 页码](q3_supplementary3_parameters.csv)",
        table(["行为","参数","补充三","本次 R 拟合"],[[r.behavior,r.term,
            f"{r.published_mean:.2f} [{r.published_lo95:.2f}, {r.published_hi95:.2f}]",
            f"{r['mean']:.2f} [{r.lo95:.2f}, {r.hi95:.2f}]"] for _,r in selected_coef.iterrows()]),
        agreement+"hu 表示零值的对数优势。抬头的 hu_CallTypeflat 为负，意味着 flat 比控制声更容易产生正的抬头时长。scaleNpenguins 描述抬头正值模型中群体规模增加一个样本标准差时的变化。接近模型中的声型主效应对应参考情境 after。",
        f"在搜寻、flat、100 m、{size:g} 只和播放前鸣叫数中位数条件下，两套模型对同一事件的预测如下。R 一栏是后验均值与 95% 可信区间，第三问一栏给出点估计与配对重抽样 95% 区间。[全部条件](q3_code1_comparison.csv)",
        table(["事件","R 基准，%","第三问主模型，%"],[[r.endpoint,
            f"{100*r.r_probability:.1f} [{100*r.r_lo95:.1f}, {100*r.r_hi95:.1f}]",
            f"{100*r.q3_probability:.1f} [{100*r.q3_lo95:.1f}, {100*r.q3_hi95:.1f}]"] for r in subset.itertuples()]),
        "Code1 的抬头和鸣叫 hurdle 部分仅含声型，同一声型的正值概率在各距离和情境相同。接近概率已对新实验的随机截距积分。两套模型的差值反映模型结构与估计方法的共同影响。",
        f"补充四描述传播距离与录音相似度。以 flat 为例，50、100、150 m 的声谱图相关系数依次为 "+
        "、".join(f"{v:.3f}" for v in flat.spectrogram_estimate)+"，相同条件下按实测情境占比综合的 60 秒抬头概率依次为 "+
        "、".join(f"{100*v:.1f}%" for v in flat.HeadsUp_p_event_60)+"。三种声学指标来自问题二复核过的 Code2 GAM，行为来自 Data02。它们按声型和距离并排展示，两项实验没有逐次回放的共同记录。[九种条件的声学与行为对照](q3_supplementary4.csv)"])
    (output/"q3_results_summary.md").write_text("\n\n".join(res)+"\n",encoding="utf-8")

    y=val.ResponseYesNo.to_numpy()
    p=val.overall_probability.to_numpy()
    auc=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(1-y))
    candidate=scores.copy()
    surv=times.copy()
    def weighted_rows(a,by):
        a["total"]=a.n*a.validation_nll
        b=a.groupby(by)[["n","total"]].sum().reset_index()
        return [[*[r[c] for c in by],f"{r.total/r.n:.4f}"] for _,r in b.iterrows()]
    rows=[]
    for h,c in zip(BEHAVIORS,["HeadsUpDuration","VocalNcalls","ApproachDuration"]):
        obs=val[c+"_observed"]; est=val[c+"_expected"]
        rows.append([h,*[f"{np.mean((val[h+f'_observed_{n}']-val[h+f'_predicted_{n}'])**2):.3f}" for n in [5,15,60]],
                     f"{(obs>0).mean():.3f}／{val[c+'_p_positive'].mean():.3f}",f"{obs.mean():.2f}／{est.mean():.2f}",
                     f"{np.abs(obs-est).mean():.2f}"])
    cal=val.assign(bin=pd.qcut(val.overall_probability,5,duplicates="drop"))
    calibration=cal.groupby("bin",observed=True).agg(n=("expID","size"),pred=("overall_probability","mean"),obs=("ResponseYesNo","mean"))
    diag=["# 分组验证、区间与模型边界",
          f"外层五折留出完整实验，每折仅在训练实验中再分五折选择总体响应结构和正鸣叫次数分布。全部 {len(val)} 行获得折外预测，且 {pairs} 对均未跨折。总体响应 Brier = {np.mean((p-y)**2):.3f}，AUC = {auc:.3f}，八类组合平均对数损失 = {val.combo_logloss.mean():.3f}。各外折选中的模型见 grouped_oof.csv。以下是内部验证，外部泛化仍需独立实验评估。",
          table(["总体响应预测分组","样本数","平均预测概率","观测率"],
                [[j+1,int(r.n),f"{r.pred:.3f}",f"{r.obs:.3f}"] for j,(_,r) in enumerate(calibration.iterrows())]),
          "## 模型选择与敏感性",
          table(["对象","候选结构","折外平均负对数似然 ↓"],weighted_rows(candidate,["endpoint","model"])),
          f"总体响应选择 {structure}，使用工作独立边际 Logistic 并对配对整组重抽样。固定斜率施加弱岭惩罚以控制近分离。随机截距 Logistic 采用 24 点 Gauss–Hermite 积分，实验截距标准差 {glmm_sd:.3f}。积分后的 control、flat、mod、tv 标准化概率依次为 "+", ".join(f"{x:.3f}" for x in glmm)+"。控制声仅一条音轨，主模型的区间条件于所用刺激集合，未估计新音轨的预测误差。",
          f"组合第二阶段为七类岭多项 Logistic。模型为三种自然声设置共有效应，惩罚强度取其他斜率的 1/10，以便在自然声间共享信息。控制声和等权自然声参考设计各加每类 0.25 个伪计数。其余斜率惩罚强度默认 2，敏感性表另给出强度 1、4 及按非空潜伏期编码事件的预测。发生率取对应行为组合的边际和。固定目标协变量分布、模型选择和情境权重后，按 {pairs} 个 expID 有放回整组抽取，取点态百分位区间。稀有组合的效应受收缩强度影响。",
          table(["行为","时间模型","折外负对数似然 ↓"],weighted_rows(surv,["behavior","model"])),
          "三种时间模型使用相同观测窗、即时响应区间与折。单分布 Weibull 和对数正态 AFT 把全体未发生记录纳入右删失似然。主模型在窗内的生存函数为 S(t)=1−p·F(t)/F(60)，p 为该行为的 60 秒发生概率，F 为对数正态分布。未响应似然为 1−p；已响应使用 p·f(t)/F(60)，零秒使用 p·F(0.5)/F(60)。接近的已响应时间合并估计。两部分建模容许大量未响应与少数早期事件并存。RMST 为 60(1−p)+p·E[T | T≤60]。仅在 0–60 秒内解释。",
          table(["行为","5 秒 Brier","15 秒 Brier","60 秒 Brier","强度正值率 实测／预测","强度均值 实测／预测","强度 MAE"],rows),
          f"Code1 的三个选定模型已用 R 重新拟合。下表的 Rhat 和 ESS 汇总固定效应、随机截距标准差及负二项形状参数，发散与树深计数覆盖全部预热后迭代。主模型的正鸣叫次数分布为 {family}。播放前次数以 log(1+x) 进入主模型的正值概率和次数两部分。",
          table(["R 基准","最大 Rhat","最小 bulk ESS","最小 tail ESS","发散","达到树深上限"],
                [[r.behavior,f"{r.rhat_max:.3f}",int(r.bulk_ess_min),int(r.tail_ess_min),
                  r.divergences,r.treedepth_hits] for r in diagnostics.itertuples()]),
          f"所有主拟合、五折验证和 {n_boot} 次配对重抽样均要求优化器成功返回；输出还校验八类概率总和、条件构成总和、事件与无事件互补、强度期望乘积以及时间边界。结果以群体为单位。"]
    head=val[val.HeadsUpDuration_observed>0]
    timing_errors=[]
    for h in BEHAVIORS:
        responders=val[val[h+"_observed_latency_s"].notna()]
        observed=responders[h+"_observed_latency_s"]
        prediction=responders[h+"_predicted_median_if_event_s"]
        errors=np.where(observed.eq(0),np.maximum(prediction-.5,0),abs(prediction-observed))
        timing_errors.append([h,len(errors),f"{np.mean(errors):.2f}",f"{np.median(errors):.2f}"])
    head_mae=[(head[c]-head.HeadsUpDuration_observed).abs().mean() for c in
              ["heads_positive_mean_nb","heads_positive_mean_bounded"]]
    diag.insert(-1,f"另以原研究采用的计数分布拟合正抬头时长。在 {len(head)} 个正观测上，零截断负二项的折外 MAE 为 {head_mae[0]:.2f} 秒，有界均值模型为 {head_mae[1]:.2f} 秒。主表采用有界均值模型，以符合观测窗口的上限。")
    diag.extend(["在留出的实际响应回放中，把预测的条件中位时间与观测秒数对照。0 秒记录按首个半秒处理，误差取预测值到该时间区间的距离。60 秒内是否发生响应的成绩见上面的 Brier 分数。",
                 table(["行为","折外实测响应数","平均绝对时间误差（秒）","绝对时间误差中位数（秒）"],timing_errors)])
    flagged=diagnostics.loc[(diagnostics.rhat_max>=1.01)|(diagnostics.bulk_ess_min<400)|
                            (diagnostics.tail_ess_min<400)|(diagnostics.divergences>0)|
                            (diagnostics.treedepth_hits>0),"behavior"]
    if len(flagged):
        diag.append("本次采样中，"+"、".join(flagged)+" 有诊断项需留意。核查条件为最大 Rhat ≥ 1.01、最小 ESS < 400、出现发散或达到树深上限。上述模型的数值应结合诊断表解读。")
    (output/"q3_model_diagnostics.md").write_text("\n\n".join(diag)+"\n",encoding="utf-8")
