#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the complete reviewed-regions list: tables (native + Chinese labels),
display formulas (native), inline-math paragraphs (assembled translation),
references (kept in English), figures (bitmap kept, reviewed)."""
import json, os, re
import fitz
from PIL import ImageFont

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
tr = json.load(open(os.path.join(JOB, 'translations.json'), encoding='utf-8'))
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
LETTERS = re.compile(r'[A-Za-z]{2,}')
CAPTION = re.compile(r'^Table\s+(\d+)\b')
TOKEN = re.compile(r'⟦M\d{4}⟧')
FONT_PATH = r'C:\Windows\Fonts\simsun.ttc'
B = {}
for p in doc['pages']:
    for b in p['blocks']:
        B[b['id']] = b


def expanded(bid):
    e = tr.get(bid, {})
    if not e or e.get('action') == 'keep':
        return None
    text = e.get('text')
    if not text:
        return None
    pm = B[bid].get('protected_math') or []
    table = {o['token']: o['source'] for o in pm}

    def sub(m):
        src = table.get(m.group(0), '')
        return ' ' + src.strip() + ' '
    text = TOKEN.sub(sub, text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(' (', '（').replace(') ', '）') if False else text
    return text


def width(text, size):
    """Estimate drawn width the way the engine's SimHei metrics do:
    full-width glyphs ~1.0 em, ASCII ~0.56 em."""
    w = 0.0
    for ch in text:
        w += size * (1.0 if ord(ch) > 127 else 0.56)
    return w


def fits(text, size, w):
    return width(text, size) <= w - 1.2


def page_blocks(page):
    H = page['height']
    return [b for b in page['blocks'] if b['bbox'][1] >= 60 and b['bbox'][3] <= H - 40]


def table_groups(page):
    blocks = page_blocks(page)
    caps = [b for b in blocks if b.get('font_size', 99) <= 9.5 and CAPTION.match(str(b.get('source') or ''))]
    groups = []
    for cap in sorted(caps, key=lambda b: b['bbox'][1]):
        y = cap['bbox'][1]
        num = CAPTION.match(cap['source']).group(1)
        band = [b for b in blocks if b is not cap and abs((b['bbox'][1] + b['bbox'][3]) / 2 - (cap['bbox'][1] + cap['bbox'][3]) / 2) < 6 and b['bbox'][0] >= cap['bbox'][0] - 40]
        cells = []
        for b in sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])):
            if b in band or b is cap:
                continue
            if b['bbox'][1] < cap['bbox'][3] - 1:
                continue
            src = str(b.get('source') or '')
            if b.get('font_size', 99) > 9.6 or b['kind'] == 'image':
                if cells and b['bbox'][1] > cells[-1]['bbox'][3]:
                    break
                continue
            if src.startswith('Figure '):
                break
            cells.append(b)
        members = band + [cap] + cells
        if len(members) < 3:
            continue
        groups.append(dict(num=int(num), caption=cap, members=members))
    return groups


# manual label overrides for merged multi-line cells: bid -> [(bbox, text, size, bold)]
OVERRIDES = {
    'p0005-b0018': [
        ([189.8, 546.2, 395.8, 555.5], '近场水平、特征距离、衰减指数、残差标准差', 7.0, False),
        ([189.8, 561.4, 390.4, 570.7], '可能识别、普通识别与可靠识别的阈值（0.3, 0.5, 0.7）', 7.0, False),
    ],
}


GLYPHS = {'\u2212': '-', '\u2217': '*', '\u02c6': '', '\u2212': '-'}


def clean(text):
    """Drop/replace characters SimHei cannot draw (labels are drawn with SimHei)."""
    for bad, good in [('\u2212', '-'), ('\u2217', '*'), ('\u02c6', ''), ('\u02d9', '')]:
        text = text.replace(bad, good)
    return text


def make_label(b, text):
    x0, y0, x1, y1 = b['bbox']
    size = float(b.get('font_size') or 7.5)
    bold = bool(b.get('bold'))
    w = x1 - x0
    text = clean(text.replace(' ± ', '±').replace('（ ', '（').replace(' ）', '）'))
    while size > 5.2 and not fits(text, size, w):
        size -= 0.2
    h = max(y1 - y0, size * 1.50)
    cy = (y0 + y1) / 2
    box = [round(x0 - 0.4, 2), round(cy - h / 2 - 0.7, 2), round(x1 + 0.4, 2), round(cy + h / 2 + 0.7, 2)]
    return dict(source=str(b.get('source')), text=text, bbox=box,
                font_size=round(size, 2), rotation=0, align='left', bold=bold)


def fit_labels(labels, right_edge):
    """Widen cramped labels to the right (single line only) instead of letting the
    engine wrap them into a second line that the one-line box cannot hold."""
    placed = sorted(labels, key=lambda l: (l['bbox'][1], l['bbox'][0]))
    for i, l in enumerate(placed):
        x0, y0, x1, y1 = l['bbox']
        w = x1 - x0
        if fits(l['text'], l['font_size'], w):
            continue
        cy = (y0 + y1) / 2
        cap = right_edge
        for other in placed[i + 1:]:
            if other['bbox'][0] <= x1 + 0.5:
                continue
            if min(other['bbox'][3], y1) - max(other['bbox'][1], y0) > 0.5:
                cap = min(cap, other['bbox'][0] - 0.8)
                break
        need = width(l['text'], 5.2) + 1.4
        new_x1 = min(cap, x0 + max(w, need))
        size = 5.2
        probe = 7.5
        while probe >= 5.2:
            if fits(l['text'], probe, new_x1 - x0):
                size = round(probe, 2)
                break
            probe -= 0.1
        h = size * 1.50
        l['font_size'] = size
        l['bbox'] = [x0, round(cy - h / 2 - 0.7, 2), round(new_x1, 2), round(cy + h / 2 + 0.7, 2)]


def table_regions():
    regions = []
    for page in doc['pages']:
        pno = page['number']
        for g in table_groups(page):
            members = g['members']
            labels = []
            src_ids = []
            for b in members:
                src_ids.append(b['id'])
                if b['id'] in OVERRIDES:
                    for box, txt, size, bold in OVERRIDES[b['id']]:
                        labels.append(dict(source=str(b.get('source')), text=txt, bbox=box,
                                           font_size=size, rotation=0, align='left', bold=bold))
                    continue
                text = expanded(b['id'])
                src = str(b.get('source') or '')
                fs = float(b.get('font_size') or 7.5)
                hgt = b['bbox'][3] - b['bbox'][1]
                if not text or not HAN.search(text) or not LETTERS.search(src):
                    continue
                if b['kind'] != 'text':
                    continue
                if hgt > 2.4 * fs:
                    # merged multi-line block without an override: keep native text
                    continue
                labels.append(make_label(b, text))
            fit_labels(labels, 505.0)
            boxes = [b['bbox'] for b in members] + [l['bbox'] for l in labels]
            ys0 = min(x[1] for x in boxes); ys1 = max(x[3] for x in boxes)
            bbox = [89.5, round(ys0 - 1.6, 2), 505.0, round(ys1 + 2.0, 2)]
            for l in labels:
                bb = l['bbox']
                l['bbox'] = [round(max(89.5, bb[0]), 2), round(max(bbox[1], bb[1]), 2),
                             round(min(505.0, bb[2]), 2), round(min(bbox[3], bb[3]), 2)]
            regions.append(dict(id=f'p{pno:04d}-table{g["num"]:02d}', kind='table', page=pno,
                                bbox=bbox, source_ids=src_ids, reviewed=True,
                                reason=f'源 Table {g["num"]} 单元格被 PDF 抽取成独立块，逐块重排会打乱行列；整表按原版式裁取，数字与符号保持原样，文字单元格用中文标签覆盖。',
                                labels=labels))
    return regions


# ---------------------------------------------------------------- formulas
FORMULAS = [
    ('p0007-eq01', 7, [183.0, 670.0, 505.0, 706.0], ['p0007-b0011', 'p0007-b0012', 'p0007-b0013', 'p0007-b0014'], '式 (1) Kruskal–Wallis 统计量'),
    ('p0009-eq02', 9, [137.0, 522.0, 505.0, 541.0], ['p0009-b0011', 'p0009-b0012', 'p0009-b0013'], '式 (2) 保真度指数定义'),
    ('p0009-eq03', 9, [242.0, 725.0, 505.0, 757.5], ['p0009-b0017', 'p0009-b0018'], '式 (3) 曲线 M3'),
    ('p0011-eq05', 11, [196.0, 600.0, 505.0, 630.0], ['p0011-b0063', 'p0011-b0064', 'p0011-b0065'], '式 (5) 识别概率'),
    ('p0012-eq06', 12, [204.0, 111.5, 505.0, 141.5], ['p0012-b0004', 'p0012-b0005'], '式 (6) 门控响应模型'),
    ('p0013-eq07', 13, [243.0, 120.5, 505.0, 151.0], ['p0013-b0005', 'p0013-b0006'], '式 (7) 距离对阈值的弹性'),
    ('p0022-eq13', 22, [207.0, 484.5, 505.0, 513.5], ['p0022-b0039', 'p0022-b0040'], '式 (13) 含噪条件'),
    ('p0023-eq14', 23, [247.0, 636.5, 505.0, 655.5], ['p0023-b0019', 'p0023-b0020'], '式 (14) 噪声下降距离'),
]


def formula_regions():
    out = []
    for rid, page, bbox, ids, why in FORMULAS:
        out.append(dict(id=rid, kind='formula', page=page, bbox=bbox, source_ids=ids, reviewed=True,
                        reason=f'{why}被抽取成多个碎片块，逐块重排会错位；整式按原版式原生裁取（不翻译）。'))
    return out


# ------------------------------------------------------- inline-math paragraphs
PARAGRAPHS = ['p0008-b0021', 'p0010-b0020', 'p0011-b0067', 'p0016-b0028', 'p0019-b0057', 'p0021-b0004']

# The inline/display maths inside these blocks was extracted as badly mangled
# fragments (bboxes straddle two printed lines and the source strings interleave
# neighbouring expressions), so native crops cannot reproduce them reliably.
# These paragraphs are therefore rendered from hand-written Chinese text in which
# every expression is spelled out linearly; no native glyph cropping is used.
PARA_TEXT = {
    'p0008-b0021': '每对显著的比较都涉及平直型鸣叫，而调制型与双声型在任何共有度量上都无法区分。仅在调制型与双声型中定义的脉冲率同样没有差异（Mann–Whitney 检验 p = 0.278）。PERMANOVA 给出伪 F = 2.69、R² = 0.049、p = 0.008，其中平直型–双声型（p = 0.005）与平直型–调制型（p = 0.036）显著，调制型–双声型不显著（p = 0.31）。这些差异真实但很小：鸣叫类型解释的多变量方差不足 5%。',
    'p0010-b0020': '对 μk(d) 设三个阈值将其分为四种状态：不可识别（μ < τP）、可能识别（τP ≤ μ < τR）、识别（τR ≤ μ < τS）与可靠识别（μ ≥ τS），其中 τP = 0.3、τR = 0.5、τS = 0.7。数值 0.5 与 0.7 是问题 4 给出的基线与上界。数值 0.3 来自噪声基底：平直型鸣叫在 256 m 处已无可用的信号，',
    'p0011-b0067': '其中 σk 为类型 k 的残差标准差，Φ 为标准正态分布函数。在 τ = 0.5 处它再现了观测到的超过阈值的行占比（平直型在 64 m 处：0.83，观测值为 0.88）。若要求 ρ = 0.9，距离缩短为 58、116 和 151 m。',
    'p0016-b0028': '全部 278 次回放纳入一个惩罚多项逻辑回归。响应为五个类别之一 m（无、仅 H、仅 V、HV、含接近），刺激为四水平因子（对照、平直型、调制型、双声型）：ln[P(M = m) / P(M = none)] = αm + βTm s + γTm z，（9）其中 s 含刺激哑变量，z 含情境、距离与 ln N。对照声音在后两个类别中计数为零，因此需要 L2 惩罚；其强度由按实验分组的 5 折交叉验证选取，区间来自对实验的自举。',
    'p0019-b0057': '表现出具有功能意义的行为，则该鸣叫是有效的。按照 5.4 节的标定，抬头与声音回应（接触）采用可能识别阈值 τP = 0.3，接近（招募）采用识别阈值 τR = 0.5。该指数为 E(k, d, N) = Σc πc Σb wb ρk(d; τb, δ) θk,b(c, N)，（11）其中 ρk(d; τ, δ) = Φ(((1 - δ) μk(d) - τ) / σk)，θk,b(c, N) 是鸣叫被识别后行为 b 的概率，wb 是该行为的功能权重，πc 是情境权重，δ 是噪声造成的相关性总体损失（基线中为 0）。该指数介于 0 与 1 之间。若一个响应组合按其各行为权重之和计分，则各组合概率的加权和等于 Σb wb P(b)，所以该指数也就是组合加权得分。',
    'p0021-b0004': '指标的每个分量都有自己的半径，即识别概率降至一半处的距离：接触半径 Rcon,k = d*k(τP)，招募半径 Rrec,k = d*k(τR)。我们还报告了距离抽样 [17] 中使用的有效面积半径 Reff,k(τ) = √( ∫ 2 r ρk(r; τ, δ) dr )（积分下限 0、上限 Rmax），（12）其中 r 为距鸣叫者的距离，Rmax = 400 m；它不依赖于截断值，在噪声下也不会骤然崩塌。对于典型群体（N = 8），接触半径分别为 155、225 和 245 m，招募半径分别为 83、144 和 177 m（表 14）。群体规模会改变指数的水平（N = 2 的近场值为 0.43–0.49，N = 20 为 0.21–0.23），但几乎不改变其半径。数据只记录群体是否作出响应。群体作出任何响应的概率为 Pg = ρk(d; τP) θk,any；作出接近的概率为 PA = ρk(d; τR) θk,A。关于响应者数量，我们考虑三种假设：S1，一只鸟响应（期望数量 Pg）；S2，各鸟独立响应（N[1 - (1 - Pg)1/N]）；S3，群体中有比例为 φ 的个体响应（φNPg）。',
}


def paragraph_regions():
    out = []
    for bid in PARAGRAPHS:
        b = B[bid]
        text = PARA_TEXT.get(bid) or (tr.get(bid, {}).get('text') or '')
        x0, y0, x1, y1 = b['bbox']
        bbox = [round(max(0.5, x0 - 1.0), 2), round(y0 - 1.0, 2),
                round(min(505.0, x1 + 1.0), 2), round(y1 + 1.0, 2)]
        out.append(dict(id=f'{bid}-para', kind='paragraph', page=b['page'], bbox=bbox,
                        source_ids=[bid], reviewed=True, translation=text,
                        native_math=[],
                        reason='该段被行内公式与显示公式切成多个碎片块，逐块重排会错位；整段作为区域重排，公式改写为线性文本以保持可读。'))
    return out



# ------------------------------------------------------------------ references
def ref_regions():
    out = []
    p25 = [bid for bid in ('p0025-b0015', 'p0025-b0016', 'p0025-b0017', 'p0025-b0018', 'p0025-b0019',
                           'p0025-b0020', 'p0025-b0021', 'p0025-b0022', 'p0025-b0023')]
    p26 = [f'p0026-b{n:04d}' for n in range(3, 35)]
    for page, ids, bbox in ((25, p25, [89.6, 583.0, 505.0, 772.5]),
                            (26, p26, [89.6, 74.0, 505.0, 764.0])):
        boxes = [B[i]['bbox'] for i in ids] + [bbox]
        ys0 = min(x[1] for x in boxes); ys1 = max(x[3] for x in boxes)
        out.append(dict(id=f'p{page:04d}-refs', kind='table', page=page,
                        bbox=[89.6, round(ys0 - 1.2, 2), 505.0, round(ys1 + 1.2, 2)],
                        source_ids=ids, reviewed=True,
                        reason='参考文献条目按学术惯例保留英文原文（作者、题名、期刊、DOI 不宜改写）；整区原生裁取。'))
    return out


# --------------------------------------------------------------------- figures
FIGURES = [
    (5, 'p0005-fig001', [106.7, 74.8, 488.5, 232.2]),
    (7, 'p0007-fig001', [119.2, 74.8, 476.1, 290.6]),
    (7, 'p0007-fig002', [100.5, 407.8, 494.8, 533.3]),
    (10, 'p0010-fig001', [144.1, 420.0, 451.2, 611.3]),
    (12, 'p0012-fig001', [100.5, 393.5, 494.8, 546.8]),
    (13, 'p0013-fig001', [119.2, 281.6, 476.1, 394.3]),
    (15, 'p0015-fig001', [90.1, 250.4, 272.7, 359.7]),
    (15, 'p0015-fig002', [305.9, 232.5, 505.1, 359.7]),
    (15, 'p0015-fig003', [193.9, 561.5, 401.4, 683.8]),
    (18, 'p0018-fig001', [106.7, 74.8, 488.5, 222.9]),
    (20, 'p0020-fig001', [100.5, 540.3, 494.8, 694.9]),
    (23, 'p0023-fig001', [106.7, 74.8, 488.5, 242.1]),
]


def figure_regions():
    out = []
    for page, rid, bbox in FIGURES:
        x0, y0, x1, y1 = bbox
        out.append(dict(id=rid, kind='figure', page=page,
                        bbox=[round(max(0.5, x0 - 2), 2), round(y0 - 2, 2),
                              round(min(505.0, x1 + 2), 2), round(y1 + 2, 2)],
                        source_ids=[], reviewed=True,
                        reason='位图为矢量/位图混合的统计图或流程示意图，图形本体原生保留，图内坐标轴、图例与框内说明为英文技术标签；图注已在正文中译为中文。'))
    return out


def main():
    regions = table_regions() + formula_regions() + paragraph_regions() + ref_regions() + figure_regions()
    json.dump(dict(regions=regions), open(os.path.join(JOB, 'regions_all.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    nl = sum(len(r.get('labels') or []) for r in regions)
    print('regions', len(regions), 'labels', nl)
    for r in regions:
        print(f"p{r['page']:02d} {r['id']:20s} {r['kind']:9s} ids={len(r.get('source_ids', [])):3d} labels={len(r.get('labels') or []):3d} bbox={r['bbox']}")


if __name__ == '__main__':
    main()
