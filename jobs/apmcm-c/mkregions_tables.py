#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build regions.json entries for tables: native table crop + Chinese labels.

Detects tables from 'Table N' caption blocks, groups the small-font cell blocks
under each caption, turns every translated text cell into a label drawn over the
native cell, and marks the covered blocks as kept so the base page keeps the
original glyphs.
"""
import json, os, re, sys
import fitz
from PIL import ImageFont

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
tr = json.load(open(os.path.join(JOB, 'translations.json'), encoding='utf-8'))
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
LETTERS = re.compile(r'[A-Za-z]{2,}')
CAPTION = re.compile(r'^Table\s+(\d+)\b')
SOURCE = fitz.open(os.path.join(JOB, 'source.pdf'))
FONT_PATH = r'C:\Windows\Fonts\simsun.ttc'

# doc json block translations live in translations.json with expanded text


def expanded(entry):
    if not entry:
        return None
    if entry.get('action') == 'keep':
        return None
    return entry.get('text')


def width(text, size):
    f = ImageFont.truetype(FONT_PATH, max(1, int(round(size * 4))))
    return f.getlength(text) / 4.0


def page_blocks(page):
    H = page['height']
    out = []
    for b in page['blocks']:
        y0, y1 = b['bbox'][1], b['bbox'][3]
        if y0 < 60 or y1 > H - 40:
            continue
        out.append(b)
    return out


def table_groups(page):
    blocks = page_blocks(page)
    caps = [b for b in blocks if b.get('font_size', 99) <= 9.5 and CAPTION.match(str(b.get('source') or ''))]
    groups = []
    for cap in sorted(caps, key=lambda b: b['bbox'][1]):
        y = cap['bbox'][1]
        num = CAPTION.match(cap['source']).group(1)
        # caption band: all blocks whose vertical centre is within the caption line
        band = [b for b in blocks if b is not cap and abs((b['bbox'][1] + b['bbox'][3]) / 2 - (cap['bbox'][1] + cap['bbox'][3]) / 2) < 6 and b['bbox'][0] >= cap['bbox'][0] - 40]
        # cells: small-font blocks below caption, stop at body/heading/figure caption
        cells = []
        for b in sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])):
            if b in band or b is cap:
                continue
            if b['bbox'][1] < cap['bbox'][3] - 1:
                continue
            src = str(b.get('source') or '')
            if b.get('font_size', 99) > 9.6 or b['kind'] == 'image':
                # body-sized block below caption ends the table
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


def make_label(b, entry_text):
    x0, y0, x1, y1 = b['bbox']
    size = float(b.get('font_size') or 7.5)
    text = entry_text.strip()
    # shrink until it fits the cell width on one line
    while size > 6.0 and width(text, size) > (x1 - x0) + 1.0:
        size -= 0.25
    h = max(y1 - y0, size * 1.30)
    cy = (y0 + y1) / 2
    box = [x0 - 0.4, round(cy - h / 2 - 0.6, 2), x1 + 0.4, round(cy + h / 2 + 0.6, 2)]
    return dict(source=str(b.get('source')), text=text, bbox=box,
                font_size=round(size, 2), rotation=0, align='left')


def main():
    regions = []
    patched = {}
    by_page = {}
    for page in doc['pages']:
        by_page[page['number']] = page
    for pno in sorted(by_page):
        page = by_page[pno]
        for g in table_groups(page):
            members = g['members']
            xs0 = min(b['bbox'][0] for b in members); ys0 = min(b['bbox'][1] for b in members)
            xs1 = max(b['bbox'][2] for b in members); ys1 = max(b['bbox'][3] for b in members)
            bbox = [89.5, round(ys0 - 1.4, 2), 506.0, round(ys1 + 1.8, 2)]
            labels = []
            src_ids = []
            for b in members:
                src_ids.append(b['id'])
                text = expanded(tr.get(b['id'], {}))
                src = str(b.get('source') or '')
                if text and HAN.search(text) and LETTERS.search(src):
                    labels.append(make_label(b, text))
            regions.append(dict(id=f'p{pno:04d}-table{g["num"]:02d}', kind='table', page=pno,
                                bbox=bbox, source_ids=src_ids, reviewed=True,
                                reason=f'源 Table {g["num"]} 单元格被抽取成独立块，段落重排会破坏行列；整表原生裁取，中文标签覆盖文字单元格，数字与公式保持原样。',
                                labels=labels))
            for b in members:
                entry = tr.get(b['id'])
                if entry:
                    patched[b['id']] = entry.get('input_hash')
    out = JOB
    if '--write-keep' in sys.argv:
        for bid, h in patched.items():
            tr[bid] = dict(input_hash=h, action='keep', keep_kind='table_cell',
                           reason='表格单元格：由表格区域原生渲染并用中文标签覆盖。')
        json.dump(tr, open(os.path.join(JOB, 'translations.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    json.dump(dict(regions=regions), open(os.path.join(JOB, 'regions_tables.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    tot = 0
    for r in regions:
        tot += len(r['labels'])
        print(f"p{r['page']:02d} {r['id'][-8:]} members={len(r['source_ids']):3d} labels={len(r['labels']):3d} bbox={r['bbox']}")
    print('regions', len(regions), 'labels', tot, 'patched', len(patched))


if __name__ == '__main__':
    main()
