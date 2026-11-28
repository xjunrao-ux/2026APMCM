import json, sys
from collections import Counter
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import pdf_translate as E

J = r'X:\2026APMCM\jobs\apmcm-c'
doc = json.load(open(J + r'\document.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}

TEXT = {
 'p0011-b0003': '其保真度的均值加两倍标准差为 0.28。按信噪比（SNR）= r2/(1 - r2)，三个阈值大致对应 -10、-5 与 0 dB。可靠阈值必须保持在近场水平以下，而这对 tv 仅勉强成立。有效距离即均值曲线与阈值相交之处，dk(τ) = ds,k[(ak/τ)2 -1]1/(2γk)，τ < ak。（4）',
 'p0014-b0035': '（通勤）和 52.8%（觅食后）则较低。情境与群体规模存在混杂（搜寻时群体规模中位数为 2，通勤时为 15），因此将四个因素一并纳入逻辑回归：ln[p/(1 - p)] = β0 + βtype + βctx + βd(d -80)/50 + βN ln(N/8)，（8）其中 p 为该行为发生的概率，βtype 与 βctx 分别为鸣叫类型和情境的效应（参照水平：平直型、通勤），d 为回放距离，N 为企鹅数量。同一音轨会被多次播放，因此标准误按音轨聚类（28 条音轨）。',
 'p0017-b0012': '的响应发生在最初 10 s 内，最慢为 17.6 s，且 35.3% 的鸣叫回放从未引发响应。这正是“无响应并不意味着未感知到刺激”在数据中的表现：无响应是独立的结果，而不是尚未发生的响应。普通生存模型将其视为删失，从而假设只要观察足够久每个群体都会响应，这在此处是错误的。因此我们采用混合治愈模型 [16]：群体以概率 π 响应，若响应，则潜伏期 T 的对数服从正态分布，P(T > t) = 1 - π + π[1 - Φ((ln t - xTβ)/σ)]，ln[π/(1 - π)] = xTα，（10）',
}
for bid, t in TEXT.items():
    a = Counter(E.NUMBER.findall(B[bid]['source']))
    c = Counter(E.NUMBER.findall(t))
    print(bid, 'src-txt', dict(a - c), 'txt-src', dict(c - a))

items = [[bid, t] for bid, t in TEXT.items()]
json.dump(dict(packet='d79a7e171f16855efb27975e44aeb149999931f1ce308cd747d70be01fe687bb', items=items),
          open(J + r'\replies\fix_inline.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('written')
