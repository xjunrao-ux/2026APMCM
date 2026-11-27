# -*- coding: utf-8 -*-
"""用 Data02 看：企鹅到底在哪些距离响应了？响应率随距离怎么变？"""
import pandas as pd
import numpy as np

DATA = r"C:\Users\34232\Documents\GitHub\2026APMCM\题目与要求\date\Data02_AfricanPenguins_playbacks_data.csv"
df = pd.read_csv(DATA)

calls = df[df.Type == "call"].copy()
controls = df[df.Type == "control"].copy()

print("== Data02 距离取值（自然声）==")
print(calls.Distance.value_counts().sort_index().to_string())

print("\n== 各距离：自然声 vs 对照 的响应率 ==")
rows = []
for d, g in calls.groupby("Distance"):
    c = controls[controls.Distance == d]
    rows.append(dict(
        Distance=d, n_call=len(g), n_control=len(c),
        call_resp=g.ResponseYesNo.mean(), ctrl_resp=c.ResponseYesNo.mean(),
        call_head=g.HeadsUpYesNo.mean(), call_vocal=g.VocalYesNo.mean(),
        call_approach=g.ApproachYesNo.mean(),
    ))
res = pd.DataFrame(rows).sort_values("Distance")
print(res.round(3).to_string(index=False))

print("\n== 各距离 × 叫声类型：自然声响应率 ==")
piv = calls.pivot_table(index="Distance", columns="CallType", values="ResponseYesNo", aggfunc="mean")
print(piv.round(3).to_string())
