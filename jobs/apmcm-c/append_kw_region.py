import json

P = r'X:\2026APMCM\jobs\apmcm-c\regions_all.json'
data = json.load(open(P, encoding='utf-8'))
regs = data['regions']
regs = [g for g in regs if g['id'] != 'p0001-keywords']
regs.append({
    "id": "p0001-keywords",
    "kind": "paragraph",
    "page": 1,
    "bbox": [90.1, 581.0, 505.0, 611.5],
    "source_ids": ["p0001-b0009", "p0001-b0010", "p0001-b0011"],
    "reviewed": True,
    "translation": "关键词：非洲企鹅　声学交流　信号退化　回放实验　混合治愈模型　有效性指数",
    "native_math": [],
    "reason": "源关键词行被抽取拆成三个横向碎块（Keywords 块、acoustic communication 块、signal degradation playback experiment 块），逐块重排会相互重叠；整行作为区域重排以保证按原顺序排成一行。",
})
data['regions'] = regs
json.dump(data, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('regions', len(regs))
