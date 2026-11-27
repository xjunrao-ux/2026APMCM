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
           paired_p,combo_p,score_p,glmm_sd,glmm,intensity):
    def interval(r,key,k=1):
        return f"{k*r[key]:.2f} [{k*r[key+'_lo']:.2f}, {k*r[key+'_hi']:.2f}]"
    natural=cont[cont["first"]=="natural_equal"].iloc[0]
    rates=d.groupby("Type").ResponseYesNo.agg(["sum","size","mean"])
    res=["# 第三问运行结果",
         f"139 个配对实验中，自然叫声后有 {rates.loc['call','sum']}/139 组出现响应（{100*rates.loc['call','mean']:.1f}%），控制声后有 {rates.loc['control','sum']}/139 组（{100*rates.loc['control','mean']:.1f}%）。只对自然声响应的有 {plus} 对，只对控制声响应的有 {minus} 对。双侧精确 McNemar 检验 p = {paired_p:.3g}。",
         f"在相同的情境、距离和群体规模分布上，将三种自然声等权平均，相对控制声的调整概率差为 {interval(natural,'risk_difference',100)} 个百分点，边际优势比为 {interval(natural,'marginal_OR')}。下文区间均为按实验配对重抽样所得的点态 95% 区间。",
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
         f"搜寻时，自然声与控制声的实测响应率相差 81.6 个百分点（36/38 对 5/38）。叫声与情境的六项交互系数联合得分检验 p = {score_p:.3g}，含完整交互项的模型也取得最低五折组外对数损失。距离增加 10 m 的平均预测概率变化为 {interval(effects.iloc[0],'risk_difference',100)} 个百分点，群体增加 1 只为 {interval(effects.iloc[1],'risk_difference',100)} 个百分点。计算仅使用变化后仍处于总体观测范围的行。三种自然声之间的两两差值区间均跨过零，现有数据不足以确定总体响应率排序。",
         "## 一次回放后会出现哪些行为组合",
         f"H 为抬头、V 为鸣叫、A 为接近。139 对实验内交换一次回放的完整组合标签，共进行 50,000 次置换。八类边际分布相同的全局检验 p = {combo_p:.5f}（蒙特卡洛下限 1/50,001）。",
         table(["组合","自然声次数","控制声次数","等权自然声相对控制声的调整概率差，百分点"],
               [[c,sum((d.Type=="call")&(d.code==j)),sum((d.Type=="control")&(d.code==j)),
                 interval(natural,"difference_"+c,100)] for j,c in enumerate(COMBOS)]),
         "自然声后最常见的两种正响应组合是仅鸣叫（29 次）和抬头并鸣叫（28 次），对照分别为 5 次和 0 次。三种自然声各自对控制声的八类调整差值与区间见 q3_contrasts.csv。仅接近（A）在全表只出现 1 次，HA 只出现 3 次。稀有组合的数值受到收缩强度影响，逐项区间未做多重比较校正。配对置换依赖配对内可交换假设，调整模型另使用每次回放的实际距离。",
         "## 群体规模为 8 只时，通常多久出现响应",
         "每格依次给出「60 秒内发生概率（%）／发生者首次响应中位秒数」。情境权重按 139 个实验确定，通勤 65/139、搜寻 38/139、摄食后 36/139。各情境的条件时间分布先按事件概率加权混合，再求中位数。接近时间合并 23 次事件估计，因此其条件中位数在各行相同，发生概率仍随条件变化。"]
    rows=[]
    for (c,dist),grp in avg.groupby(["CallType","Distance"],sort=False):
        if c=="control":
            continue
        g=grp.set_index("behavior")
        rows.append([c,dist,*[f"{100*g.loc[h,'p_event_60']:.1f}／{g.loc[h,'median_if_event_s']:.1f}" for h in BEHAVIORS]])
    res.extend([table(["叫声","距离 m","抬头 %／秒","鸣叫 %／秒","接近 %／秒"],rows),
                "这是模型对同条件新实验的预测。完整潜伏期表附有确切条件的样本数及事件数。n_exact=0 表示该条件缺乏直接观测；n_exact>0 且 events_exact=0 标为无响应。控制声的 139 次回放均未观察到接近，其条件潜伏期留空。mod 在 after 情境只观测到 120 m，150 m 的该情境预测含外推，综合估计也包含这一分量。未观察到行为不能判断企鹅是否听到了声音。",
                "## 响应强度",
                "下表选取 searching、100 m、8 只、播放前 60 秒鸣叫数为样本中位数的情境。每格依次为「发生正值的概率（%）／正值条件均值／每次回放的期望」。",
                table(["叫声","抬头（%／秒／秒）","鸣叫（%／次／次）","接近（%／秒／秒）"],
                      [[c,*[f"{100*r.p_positive:.1f}／{r.mean_if_positive:.1f}／{r.expected_all:.2f}"
                             for _,r in intensity[(intensity.CallType==c)&(intensity.Distance==100)&
                                                 (intensity.Context=="searching")].iterrows()]] for c in CALLS]),
                "接近正持续时间仅 23 条，表中正值时长使用合并均值，控制声的接近强度属于跨组借力的探索估计。原始鸣叫次数包含播放前已有鸣叫活动的群体，故其正值概率与鸣叫响应概率可不同。",
                "## 复现与第四问接口",
                "本机在第三问文件夹使用 `D:\\Python\\python.exe run_q3.py` 实跑。其他环境使用 `python run_q3.py`，依赖 NumPy、pandas 和 SciPy。默认按 expID 重抽样 499 次，随机种子 20261127。输出目录为 results。",
                "第四问读取 q3_behavior_combo_predictions.csv 的 p_all 即可取得一次群体回放的八类概率。q3_latency_predictions.csv 的 Context=standardized 为情境综合行，其余为分情境行。q3_overall_response_predictions.csv 与 q3_intensity_predictions.csv 给出总体响应概率和三项强度。列名后缀 _lo、_hi 对应点态 95% 区间。频数、置信区间和群体规模均以群体回放为解释单位。"])
    (OUT/"q3_results_summary.md").write_text("\n\n".join(res)+"\n",encoding="utf-8")

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
          f"固定五折，每折一次留出完整实验。全部 278 行均有折外预测，且 139 对均未跨折。总体响应 Brier = {np.mean((p-y)**2):.3f}，AUC = {auc:.3f}；八类组合平均对数损失 = {val.combo_logloss.mean():.3f}。模型选择与内部检验使用同一组折，性能需在独立实验中复核。",
          table(["总体响应预测分组","样本数","平均预测概率","观测率"],
                [[j+1,int(r.n),f"{r.pred:.3f}",f"{r.obs:.3f}"] for j,(_,r) in enumerate(calibration.iterrows())]),
          "## 模型选择与敏感性",
          table(["对象","候选结构","折外平均负对数似然 ↓"],weighted_rows(candidate,["endpoint","model"])),
          f"总体响应选择 {structure}，使用工作独立边际 Logistic 并对配对整组重抽样。固定斜率施加弱岭惩罚以控制近分离。随机截距 Logistic 采用 24 点 Gauss–Hermite 积分，实验截距标准差 {glmm_sd:.3f}。积分后的 control、flat、mod、tv 标准化概率依次为 "+", ".join(f"{x:.3f}" for x in glmm)+"。控制声仅一条音轨，主模型的区间条件于所用刺激集合，未估计新音轨的预测误差。",
          "组合第二阶段为七类岭多项 Logistic。模型为三种自然声设置共有效应，惩罚强度取其他斜率的 1/10，以便在自然声间共享信息。控制声和等权自然声参考设计各加每类 0.25 个伪计数。其余斜率惩罚强度默认 2，敏感性表另给出强度 1、4 及按非空潜伏期编码事件的预测。发生率取对应行为组合的边际和。固定目标协变量分布、模型选择和情境权重后，按 139 个 expID 有放回整组抽取，取点态百分位区间。单次出现的组合尤其难以精确估计。",
          table(["行为","时间模型","折外负对数似然 ↓"],weighted_rows(surv,["behavior","model"])),
          "三种时间模型使用相同观测窗、即时响应区间与折。单分布 Weibull 和对数正态 AFT 把全体未发生记录纳入右删失似然。主模型在窗内的生存函数为 S(t)=1−p·F(t)/F(60)，p 为该行为的 60 秒发生概率，F 为对数正态分布。未响应似然为 1−p；已响应使用 p·f(t)/F(60)，零秒使用 p·F(0.5)/F(60)。接近的已响应时间合并估计。两部分建模容许大量未响应与少数早期事件并存。RMST 为 60(1−p)+p·E[T | T≤60]。仅在 0–60 秒内解释。",
          table(["行为","5 秒 Brier","15 秒 Brier","60 秒 Brier","强度正值率 实测／预测","强度均值 实测／预测","强度 MAE"],rows),
          f"数据和原 Code1 的分组频数、抬头／接近与鸣叫描述量进行了对照。强度部分对正鸣叫次数比较零截断 Poisson 与零截断负二项，主模型为 {family}。播放前次数以 log(1+x) 进入正值概率和次数两部分。抬头秒数以分数 Logistic 估计 60 秒以内的条件均值；接近正时长取合并均值。与 Supplementary 3 的贝叶斯 hurdle NB、hurdle Poisson 和 Bernoulli 属于不同的拟合方法，本次未重跑原研究的 Stan 链。",
          f"所有主拟合、五折验证和 {n_boot} 次配对重抽样均要求优化器成功返回；输出还校验八类概率总和、条件构成总和、事件与无事件互补、强度期望乘积以及时间边界。结果以群体为单位。"]
    head=val[val.HeadsUpDuration_observed>0]
    head_mae=[(head[c]-head.HeadsUpDuration_observed).abs().mean() for c in
              ["heads_positive_mean_nb","heads_positive_mean_bounded"]]
    diag.insert(-1,f"另以原研究采用的计数分布拟合正抬头时长。在 {len(head)} 个正观测上，零截断负二项的折外 MAE 为 {head_mae[0]:.2f} 秒，有界均值模型为 {head_mae[1]:.2f} 秒。主表采用有界均值模型，以符合观测窗口的上限。")
    (OUT/"q3_model_diagnostics.md").write_text("\n\n".join(diag)+"\n",encoding="utf-8")
