import json

J = r'X:\2026APMCM\jobs\apmcm-c'
path = J + r'\regions_all.json'
data = json.load(open(path, encoding='utf-8'))
regs = data['regions']

PARA = {
 'p0011-b0003': ('p0011-eq4-text', 11, [90.1, 77.5, 505.0, 176.8],
    '其保真度的均值加两倍标准差为 0.28。按信噪比（SNR）= r2/(1 − r2)，三个阈值大致对应 −10、−5 与 0 dB。可靠阈值必须保持在近场水平以下，而这对 tv 仅勉强成立。有效距离即均值曲线与阈值相交之处，dk(τ) = ds,k[(ak/τ)2 −1]1/(2γk)，τ < ak。（4）'),
 'p0014-b0035': ('p0014-eq8-text', 14, [90.1, 385.6, 505.0, 502.8],
    '（通勤）和 52.8%（觅食后）则较低。情境与群体规模存在混杂（搜寻时群体规模中位数为 2，通勤时为 15），因此将四个因素一并纳入逻辑回归：ln[p/(1 − p)] = β0 + βtype + βctx + βd(d −80)/50 + βN ln(N/8)，（8）其中 p 为该行为发生的概率，βtype 与 βctx 分别为鸣叫类型和情境的效应（参照水平：平直型、通勤），d 为回放距离，N 为企鹅数量。同一音轨会被多次播放，因此标准误按音轨聚类（28 条音轨）。'),
 'p0017-b0012': ('p0017-eq10-text', 17, [90.1, 488.5, 505.0, 616.9],
    '的响应发生在最初 10 s 内，最慢为 17.6 s，且 35.3% 的鸣叫回放从未引发响应。这正是“无响应并不意味着未感知到刺激”在数据中的表现：无响应是独立的结果，而不是尚未发生的响应。普通生存模型将其视为删失，从而假设只要观察足够久每个群体都会响应，这在此处是错误的。因此我们采用混合治愈模型 [16]：群体以概率 π 响应，若响应，则潜伏期 T 的对数服从正态分布，P(T > t) = 1 − π + π[1 − Φ((ln t − x⊤β)/σ)]，ln[π/(1 − π)] = x⊤α，（10）'),
}

existing = {r['id'] for r in regs}
for bid, (rid, page, bbox, text) in PARA.items():
    if rid in existing:
        continue
    regs.append(dict(id=rid, page=page, kind='paragraph', bbox=bbox, source_ids=[bid],
                     reviewed=True, native_math=[], translation=text,
                     reason='行内公式被抽取为跨行碎片，原生裁字会错位；此处整段重排为线性化中文，公式以文本线性形式给出。'))

# table caption / footnote label patches
CAP = {
 'p0008-table04': dict(bbox=[126.7, 476.6, 505.0, 502.5], text='两层分类器的性能（第二层采用重复分层 5 折交叉验证）', font_size=9.0),
 'p0016-table11': dict(bbox=[130.7, 544.5, 505.0, 570.5], text='惩罚多项模型给出的五类响应预测概率（通勤，80 m，N = 8）', font_size=9.0),
}
for r in regs:
    if r['id'] in CAP:
        spec = CAP[r['id']]
        for label in r.get('labels', []):
            if abs(label['bbox'][1] - spec['bbox'][1]) < 12 or label['bbox'][3] <= spec['bbox'][3] and label['bbox'][1] >= spec['bbox'][1] - 1:
                label['bbox'] = spec['bbox']
                label['text'] = spec['text']
                label['font_size'] = spec['font_size']
                break
        else:
            r.setdefault('labels', []).append(dict(bbox=spec['bbox'], text=spec['text'], font_size=spec['font_size']))
    if r['id'] == 'p0019-table13':
        have = any('No Response' in l['text'] for l in r.get('labels', []))
        if not have:
            r.setdefault('labels', []).append(dict(bbox=[89.5, 303.0, 505.0, 328.0],
                text='† 该条件下原始数据中未观察到响应（“No Response”）；模型值从其他距离借用信息，潜伏期读作“若发生响应”。',
                font_size=7.5))

json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('regions', len(regs), 'labels', sum(len(r.get('labels', [])) for r in regs))
for r in regs:
    if r['id'] in list(PARA.values())[0] + ('p0008-table04', 'p0016-table11', 'p0019-table13'):
        pass
for r in regs:
    if r['id'] in ('p0008-table04', 'p0016-table11', 'p0019-table13'):
        print(r['id'], [(round(l['bbox'][0], 1), round(l['bbox'][1], 1), l['text'][:22]) for l in r.get('labels', [])][:3])
