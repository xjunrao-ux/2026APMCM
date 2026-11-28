import sys, json, re
from collections import Counter
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import fitz
import pdf_translate as E

J = r'X:\2026APMCM\jobs\apmcm-c'
doc = json.load(open(J + r'\document.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}
src = fitz.open(J + r'\source.pdf')

TOK = {
 'p0011-b0003': [
    ('⟦M0001⟧', [464.04, 78.39, 472.20, 88.89], '−'),
    ('⟦M0002⟧', [244.93, 94.77, 282.16, 105.46], '−10, −5'),
    ('⟦M0003⟧', [296.67, 158.89, 312.42, 169.58], '−1'),
 ],
 'p0014-b0035': [
    ('⟦M0001⟧', [202.14, 442.84, 210.30, 453.34], '−'),
    ('⟦M0002⟧', [336.75, 428.53, 357.00, 439.23], '−80'),
 ],
}
TEXT = {
 'p0011-b0003': '其保真度的均值加两倍标准差为 0.28。按信噪比（SNR）= r2/(1 ⟦M0001⟧r2)，三个阈值大致对应 ⟦M0002⟧和 0 dB。可靠阈值必须保持在近场水平以下，而这对 tv 仅勉强成立。有效距离即均值曲线与阈值相交之处，dk(τ) = ds,k[(ak/τ)2 ⟦M0003⟧]1/(2γk)，τ < ak。（4）',
 'p0014-b0035': '（通勤）和 52.8%（觅食后）则较低。情境与群体规模存在混杂（搜寻时群体规模中位数为 2，通勤时为 15），因此将四个因素一并纳入逻辑回归：ln[p/(1 ⟦M0001⟧p)] = β0 + βtype + βctx + βd(d ⟦M0002⟧)/50 + βN ln(N/8)，（8）其中 p 为该行为发生的概率，βtype 与 βctx 分别为鸣叫类型和情境的效应（参照水平：平直型、通勤），d 为回放距离，N 为企鹅数量。同一音轨会被多次播放，因此标准误按音轨聚类（28 条音轨）。',
}

for bid, toks in TOK.items():
    b = B[bid]
    clip = fitz.Rect(*b['bbox'])
    raw = src[b['page'] - 1].get_text('rawdict', flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES, clip=clip)
    objs = [fitz.Rect(t[1]) for t in toks]
    lines = []
    for blk in raw['blocks']:
        for line in blk.get('lines', []):
            chars = [c for s in line['spans'] for c in s['chars']]
            txt = ''.join(c['c'] for c in chars if clip.contains(fitz.Point((c['bbox'][0] + c['bbox'][2]) / 2, (c['bbox'][1] + c['bbox'][3]) / 2)) and not any(o.contains(fitz.Point((c['bbox'][0] + c['bbox'][2]) / 2, (c['bbox'][1] + c['bbox'][3]) / 2)) for o in objs))
            lines.append(txt)
    original = '\n'.join(lines)
    target = re.sub(r'⟦M\d{4}⟧', '', TEXT[bid])
    a = Counter(E.NUMBER.findall(re.sub(r'(?<=\d)\s*[−–-]\s*(?=\d)', '−', original)))
    c = Counter(E.NUMBER.findall(re.sub(r'(?<=\d)\s*[−–-]\s*(?=\d)', '−', target)))
    print('=====', bid)
    print('src nums', dict(a))
    print('txt nums', dict(c))
    print('diff a-c', dict(a - c), 'c-a', dict(c - a))
