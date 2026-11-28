"""Render the consolidated numerical tables and conclusions from its JSON output."""

import argparse
import json
from pathlib import Path


SOUNDS = ("control", "flat", "mod", "tv")
ENDPOINTS = ("HeadsUp","Vocal","Approach","Any")
CN = {"control":"对照","HeadsUp":"抬头","Vocal":"鸣叫","Approach":"靠近","Any":"任一响应",
      "none":"无响应","A":"仅靠近","V":"仅鸣叫","VA":"鸣叫＋靠近","H":"仅抬头",
      "HA":"抬头＋靠近","HV":"抬头＋鸣叫","HVA":"三种都出现"}
CONTRASTS = ("flat－对照","mod－对照","tv－对照","自然声等权－对照",
             "mod－flat","tv－flat","tv－mod")


def table(headers, rows):
    rule = " | ".join("---" for _ in headers)
    return ["| "+" | ".join(headers)+" |",f"| {rule} |",
            *["| "+" | ".join(map(str,row))+" |" for row in rows],""]


def ci(numbers, factor=1, signed=False):
    p,lo,hi = (factor*x for x in numbers)
    return f"{p:+.2f} [{lo:.2f}, {hi:.2f}]" if signed else f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


def support(a,exact=False):
    k = "exact" if exact else "distance"
    return f"{a[k+'_events']}/{a[k+'_sample']}" if a[k+"_sample"] else "—"


def time_cell(a):
    return ci(a["median"]) if a["median"] is not None else "—"


def observed_label(a):
    if a["status"]=="no_response_exact":
        return f"（8 只实测无响应 {support(a,True)}）"
    if a["status"]=="no_response_at_distance":
        return f"（同距实测无响应 {support(a)}）"
    return ""


def render(data):
    e = data["estimates"]
    raw = data["experiment"]
    selected = data["occurrence_selection"]
    counts = raw["combination_counts"]
    rows = data["scenarios"]
    effects = e["response_contrasts"]
    combo_effects = e["combo_contrasts"]
    risk = effects[3][0]
    strong = [CN[c] for c,n in zip(data["combo_labels"][1:],combo_effects[3][1:]) if n[1]>0]
    rare = [CN[c] for c,n in zip(data["combo_labels"][1:],combo_effects[3][1:]) if c in ("A","HA","VA") and n[1]<=0<=n[2]]
    out = [
        "# C 题第三问合并计算结果","",
        "本页由计算输出自动生成。四部分的行为发生概率来自同一个模型；强度另按各指标是否取正值分析。"
        f"所有区间以 {data['repeats']:,} 次录音轨道整组重抽样计算，每次保留轨道内的完整实验对。","",
        "## 1. 总体响应","",
        f"139 对实验中，自然声 {raw['response_natural']}/139（{raw['response_natural']/139:.2%}），"
        f"对照声 {raw['response_control']}/139（{raw['response_control']/139:.2%}）。"
        f"仅自然声响应 {raw['only_natural']} 对，仅对照响应 {raw['only_control']} 对。"
        f"精确 McNemar 检验双侧 \(p={raw['mcnemar_p']:.3g}\)。","",
    ]
    out += table(("声型","实测有响应/回放数","调整后概率 % [95% 区间]"),[
        (CN.get(s,s),f"{sum(counts[i][1:])}/{sum(counts[i])}",ci(e["standardized_response"][i][0],100))
        for i,s in enumerate(SOUNDS)])
    out += [
        f"三类自然声等权相对对照的调整差为 **{ci(risk,100,True)} 个百分点**。"
        "这一点估计等于行为组合表中无响应概率差的相反数。两张表的同时区间对应各自的比较族。","",
        "## 2. 四因素与三种行为","",
        "下表为概率差，单位为百分点，方括号内为 95% 同时区间。","",
    ]
    out += table(("声型比较","任一响应","抬头","鸣叫","靠近"),[
        (s,*[ci(a,100,True) for a in effects[i]]) for i,s in enumerate(CONTRASTS)])
    groups = ("对照声","自然声等权","自然声与对照差值的变化")
    changes = ("搜寻－赶路","进食后－赶路","100 m－50 m","8 只－4 只")
    out += table(("条件变化","播放组","任一响应","抬头","鸣叫","靠近"),[
        (change,group,*[ci(a,100,True) for a in e["what_changes"][i][j]])
        for i,change in enumerate(changes) for j,group in enumerate(groups)])
    out += [f"情境在共同覆盖的 {data['context_support_n']} 条记录上标准化，"
            "其余协变量取相同的参照分布。下表给出各情境下自然声等权减对照的概率差。",""]
    out += table(("情境","任一响应","抬头","鸣叫","靠近"),[
        (s,*[ci(a,100,True) for a in e["context_gaps"][i]])
        for i,s in enumerate(("赶路","搜寻","进食后"))])
    out += ["## 3. 单次回放的八种行为组合","",
            f"原始八类分布的总变差距离为 **{raw['combination_tv']:.3f}**。"
            f"按实验对交换标签 {raw['combination_permutations']:,} 次，"
            f"蒙特卡洛置换 \(p={raw['combination_permutation_p']:.5f}\)。",""]
    out += table(("组合","自然声次数","对照次数","对照 %","flat %","mod %","tv %",
                  "自然声等权减对照 pp [95% 同时区间]"),[
        (CN[c],sum(arm[i] for arm in counts[1:]),counts[0][i],
         *[f"{100*a[i][0]:.2f}" for a in e["standardized_combos"]],ci(combo_effects[3][i],100,True))
        for i,c in enumerate(data["combo_labels"])])
    out += table(("组合","flat－对照 pp","mod－对照 pp","tv－对照 pp"),[
        (CN[c],*[ci(combo_effects[j][i],100,True) for j in range(3)])
        for i,c in enumerate(data["combo_labels"])])
    out += [
        "自然声等权比较中，95% 同时区间全高于零的组合为**"+("、".join(strong) or "无")+"**。"
        +("以下靠近组合的区间包含零："+ "、".join(rare)+"。" if rare else ""),"",
        "## 4. 首次响应和潜伏期","",
        f"企鹅数量固定为 **{data['group_size']:g} 只**。按原始实验的情境占比综合，"
        "赶路 65/139、搜寻 38/139、进食后 36/139。模型预测和 60 秒窗内的实测状态分别显示。","",
        "### 三种行为的代表性预测","",
        "每格依次给出模型的 **60 秒内发生概率（%）／有该行为时的中位潜伏期（秒）**，"
        "方括号为逐项 95% 区间。括号中的实测无响应表示实际观察次数为零。"
        "对照声全部 139 次回放都未见靠近，其条件时间用“—”表示。","",
    ]
    out += table(("声型","距离 m","首次抬头","首次鸣叫","首次靠近"),[
        (CN.get(r["sound"],r["sound"]),r["distance"],
         *[ci(r["response"][b]["p"],100)+" ／ "+time_cell(r["response"][b])+observed_label(r["response"][b])
           for b in ENDPOINTS[:3]])
        for r in rows])
    out += [
        "`mod` 在 100 m 的三个行为实测均为 0/5，其中 8 只实测均为 0/1。"
        "表里的秒数是全数据拟合所得的条件预测。150 m 无恰好 8 只的记录，该列数值为模型预测。","",
        "### 任一行为首次响应的补充预测","",
        "实测列为同声型和同距离的响应次数／回放次数。8 只列进一步要求群体大小匹配，“—”表示缺少匹配记录。","",
    ]
    out += table(("声型","距离 m","实测","实测 8 只","响应概率 %","条件中位时间 s","限制平均等待时间 s"),[
        (CN.get(r["sound"],r["sound"]),r["distance"],
         support(a)+("（无响应）" if a["distance_sample"] and not a["distance_events"] else ""),support(a,True),
         ci(a["p"],100),time_cell(a),ci(a["rmst"]))
        for r in rows for a in [r["response"]["Any"]]])
    out += ["限制平均等待时间在 60 秒处截尾，窗内无事件的回放贡献 60 秒。",""]
    out += table(("声型","150 m－50 m 的任一响应条件中位时间差（秒）"),[
        (CN.get(s,s),ci(e["any_change_50_to_150"][i],1,True)) for i,s in enumerate(SOUNDS)])
    out += [
        "模型条件中位潜伏期与实测状态分别保留在 JSON 中。","",
        "## 5. 响应强度","",
        f"每格为正值概率 % ／ 正值条件平均强度 ／ 每次回放平均强度，方括号为逐项 95% 区间。"
        f"企鹅数量同样固定为 8 只，播前鸣叫取其中位数 {data['strength_baseline']:g} 次。"
        "三类强度的正值记录数依次为 "+ "、".join(map(str,data["strength_positive_observed"]))+"。"
        "强度中的播放后鸣叫次数可以包含原有发声活动；行为组合里的鸣叫事件采用播放后高于播放前的定义。","",
    ]
    out += table(("声型","距离 m","抬头时长（秒）","播放后鸣叫次数","靠近时长（秒）"),[
        (CN.get(r["sound"],r["sound"]),r["distance"],
         *[" ／ ".join(ci(v,100 if k==0 else 1) for k,v in enumerate(amount))
           if j!=2 or r["sound"]!="control" else ci(amount[0],100)+" ／ 无靠近实测 ／ —"
           for j,amount in enumerate(r["strength"])])
        for r in rows])
    distance = e["any_change_50_to_150"]
    context_total,context_voice = e["what_changes"][0][2][0],e["what_changes"][0][2][2]
    inc = [CN.get(s,s) for i,s in enumerate(SOUNDS) if distance[i][1]>0]
    out += ["## 6. 计算结论","",
            f"- **总体与分项。** 自然声等权响应概率比对照高 {100*risk[0]:.2f} 个百分点。"
            f"抬头、鸣叫和靠近分别高 {100*effects[3][1][0]:.2f}、{100*effects[3][2][0]:.2f}、"
            f"{100*effects[3][3][0]:.2f} 个百分点。三项同时区间均高于零。","",
            f"- **情境。** 搜寻相对赶路，自然声与对照的总体响应差扩大 "
            f"{ci(context_total,100,True)} 个百分点，鸣叫差扩大 {ci(context_voice,100,True)} 个百分点。"
            "比较对应共同协变量覆盖范围。","",
            "- **组合。** 自然声等权比较中，"+"、".join(strong)+"的增加得到同时区间支持。"
            "稀有靠近组合仍有较大不确定性。","",
            "- **潜伏期。** 从 50 m 到 150 m，"+"、".join(inc)+"的首个行为预测中位潜伏期延长，"
            "这是在窗内有响应这一条件下的结果。主表保留实测无响应。","",
            "- **强度。** 响应强度的正值概率、条件均值及单次回放期望分别报告。"
            "发声基线和是否出现鸣叫响应的定义在解释次数时均需保留。","",
            "## 7. 计算核验和复现","",
            f"在同一套留录音轨道验证中，任一响应选中 `{selected[0]['structure']}`"
            f"（对数损失 {selected[0]['gate_nll']:.4f}），"
            f"条件组合选中 `{selected[1]['structure']}`"
            f"（有响应回放的条件对数损失 {selected[1]['mix_nll_given_event']:.4f}）。"
            f"鸣叫正次数选中 `{data['count_family']}`。验证同时用于模型选择。",""]
    out += table(("潜伏期终点","有效事件","选定时间项","选定验证损失","仅截距验证损失"),[
        (CN[b],t["events"],"四因素" if t["scale"] else "仅截距",
         f"{min(t['cv']['conditional_log_loss']):.4f}",f"{t['cv']['conditional_log_loss'][0]:.4f}")
        for b,t in zip(ENDPOINTS,data["timing_selection"])])
    source = data["code1_reference"]
    if source is not None:
        out += ["**Code1 同数据复核。** 下表引用队友代码包导出的 R 诊断。"
                "同批数据的原发表参数与复拟结果也随 JSON 保存。",""]
        out += table(("行为","最大 Rhat","最小 bulk ESS","发散"),[
            (CN[r["behavior"]],f"{float(r['rhat_max']):.4f}",f"{float(r['bulk_ess_min']):.0f}",r["divergences"])
            for r in source["r_diagnostics"]])
        if "benchmark" in source:
            bench = source["benchmark"]
            labels = {"HeadsUp":"抬头时长为正","Vocal":"播放后鸣叫次数为正","Approach":"靠近发生"}
            out += ["同条件数值核对固定搜寻、flat、100 m、8 只及播前鸣叫 0 次。"
                    "R 区间为队友导出的后验可信区间。这一核对基于同一批 Data02。",""]
            out += table(("事件","Code1 概率 % [95% 可信区间]","合并版点估计 %"),[
                (labels[r["behavior"]],ci([float(r[k]) for k in ("r_probability","r_lo95","r_hi95")],100),
                 f"{100*bench['merged_point'][r['behavior']]:.2f}") for r in bench["rows"]])
        out += ["自然录音按轨道留出和重抽样，对照声仅有一条音轨，区间条件于该对照刺激。",""]
    out += ["在工作区根目录运行以下命令。带 `--teammate-zip` 时一并读入已有的 Code1 复核数据。","",
            "```powershell",
            "python -B _q3_analysis/q3_merged.py _q3_analysis/q3_merged_results.json --bootstrap 3000",
            "python -B _q3_analysis/q3_merged_tables.py _q3_analysis/q3_merged_results.json C题第三问_合并计算结果.md",
            "```","",
            "[拟合程序](_q3_analysis/q3_merged.py) · [全部数值、参数、误差区间与预测支持](_q3_analysis/q3_merged_results.json)",""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    report = render(json.loads(args.results.read_text(encoding="utf-8")))
    args.output.write_text(report,encoding="utf-8")


if __name__=="__main__":
    main()
