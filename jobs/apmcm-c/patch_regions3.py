import json

J = r'X:\2026APMCM\jobs\apmcm-c'
path = J + r'\regions_all.json'
data = json.load(open(path, encoding='utf-8'))
regs = [r for r in data['regions'] if r['id'] not in ('p0011-eq4-text', 'p0014-eq8-text', 'p0017-eq10-text')]

T11 = [
 dict(token='⟦M0001⟧', source='−', bbox=[464.04, 78.39, 472.20, 88.89], baseline=86.79, source_size=10.5),
 dict(token='⟦M0002⟧', source='−10, −5', bbox=[244.93, 94.77, 282.16, 105.46], baseline=103.17, source_size=10.5),
 dict(token='⟦M0003⟧', source='−1', bbox=[296.67, 158.89, 312.42, 169.58], baseline=167.287, source_size=10.5),
]
T14 = [
 dict(token='⟦M0001⟧', source='−', bbox=[202.14, 442.84, 210.30, 453.34], baseline=450.84, source_size=10.5),
 dict(token='⟦M0002⟧', source='−80', bbox=[336.75, 428.53, 357.00, 439.23], baseline=436.53, source_size=10.5),
]

regs.append(dict(id='p0011-eq4-text', page=11, kind='paragraph', bbox=[90.1, 77.5, 505.0, 176.8],
    source_ids=['p0011-b0003'], reviewed=True, native_math=T11,
    translation='其保真度的均值加两倍标准差为 0.28。按信噪比（SNR）= r2/(1 ⟦M0001⟧r2)，三个阈值大致对应 ⟦M0002⟧和 0 dB。可靠阈值必须保持在近场水平以下，而这对 tv 仅勉强成立。有效距离即均值曲线与阈值相交之处，dk(τ) = ds,k[(ak/τ)2 ⟦M0003⟧]1/(2γk)，τ < ak。（4）',
    reason='行内公式被抽取为跨行碎片；减号与上下标位置以原生字形裁取保留，其余文字重排为中文。'))
regs.append(dict(id='p0014-eq8-text', page=14, kind='paragraph', bbox=[90.1, 385.6, 505.0, 502.8],
    source_ids=['p0014-b0035'], reviewed=True, native_math=T14,
    translation='（通勤）和 52.8%（觅食后）则较低。情境与群体规模存在混杂（搜寻时群体规模中位数为 2，通勤时为 15），因此将四个因素一并纳入逻辑回归：ln[p/(1 ⟦M0001⟧p)] = β0 + βtype + βctx + βd(d ⟦M0002⟧)/50 + βN ln(N/8)，（8）其中 p 为该行为发生的概率，βtype 与 βctx 分别为鸣叫类型和情境的效应（参照水平：平直型、通勤），d 为回放距离，N 为企鹅数量。同一音轨会被多次播放，因此标准误按音轨聚类（28 条音轨）。',
    reason='行内公式被抽取为跨行碎片；减号以原生字形裁取保留，其余文字重排为中文。'))
regs.append(dict(id='p0017-eq10-text', page=17, kind='paragraph', bbox=[90.1, 488.5, 505.0, 616.9],
    source_ids=['p0017-b0012'], reviewed=True, native_math=[],
    translation='的响应发生在最初 10 s 内，最慢为 17.6 s，且 35.3% 的鸣叫回放从未引发响应。这正是“无响应并不意味着未感知到刺激”在数据中的表现：无响应是独立的结果，而不是尚未发生的响应。普通生存模型将其视为删失，从而假设只要观察足够久每个群体都会响应，这在此处是错误的。因此我们采用混合治愈模型 [16]：群体以概率 π 响应，若响应，则潜伏期 T 的对数服从正态分布，P(T > t) = 1 - π + π[1 - Φ((ln t - xTβ)/σ)]，ln[π/(1 - π)] = xTα，（10）',
    reason='行内公式被抽取为跨行碎片；整段以线性化中文重排，转置符号以 T 表示。'))

data['regions'] = regs
json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('regions', len(regs))
