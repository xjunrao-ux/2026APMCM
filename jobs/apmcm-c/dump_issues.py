#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""List language/math/number/structure/glyph issues with block details."""
import json, os, collections

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
qaj = json.load(open(os.path.join(JOB, 'output', 'qa.json'), encoding='utf-8'))
tr = json.load(open(os.path.join(JOB, 'translations.json'), encoding='utf-8'))

blocks = {}
for page in doc['pages']:
    for b in page['blocks']:
        blocks[b['id']] = (page['number'], b)

trans = {}
for k, v in tr.items():
    if isinstance(v, dict) and 'items' not in v:
        trans[k] = v
    elif isinstance(v, dict):
        for bid, t in v.get('items', []):
            trans[bid] = t

for it in qaj['issues']:
    bid = it['block_id']
    if not bid:
        continue
    pg, b = blocks.get(bid, (None, None))
    if not b:
        continue
    bb = ','.join(f'{v:.0f}' for v in b['bbox'])
    src = str(b.get('source') or '').replace('\n', '\\n')[:70]
    t = b.get('translation')
    if t is None:
        t = trans.get(bid)
    t = str(t or '').replace('\n', '\\n')[:70]
    print(f"{it['id'][:22]:24} pg{pg:02d} {bb:26} fs={b.get('font_size',0):4.1f} {b.get('kind')[:5]:5} act={str(b.get('action')):9} SRC[{src}] TR[{t}]")
