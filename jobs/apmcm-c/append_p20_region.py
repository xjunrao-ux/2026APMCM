import json

P = r'X:\2026APMCM\jobs\apmcm-c\regions_all.json'
data = json.load(open(P, encoding='utf-8'))
regs = [g for g in data['regions'] if g['id'] != 'p0020-inline-formula']
regs.append({
    "id": "p0020-inline-formula",
    "kind": "paragraph",
    "page": 20,
    "bbox": [88.0, 67.5, 505.0, 110.0],
    "source_ids": ["p0020-b0002", "p0020-b0004", "p0020-b0005"],
    "reviewed": True,
    "translation": "[18]，w(i) = 1/K ∑_{j=i}^{K} 1/j；当 K = 3 时得到 0.611（接近）、0.278（声音回应）和 0.111（抬头）。情境权重 πc 是实验中三种情境所占的份额。",
    "native_math": [],
    "reason": "该行正文内含行内分式公式，抽取后公式块与左右两段文字块相互重叠；整行作为区域重排，公式改写为线性文本以保持可读。",
})
data['regions'] = regs
json.dump(data, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('regions', len(regs))
