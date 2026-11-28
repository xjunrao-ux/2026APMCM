#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarise qa blockers with block geometry + translation state."""
import json, sys, collections, os

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
qaj = json.load(open(os.path.join(JOB, 'output', 'qa.json'), encoding='utf-8'))

blocks = {}
for page in doc['pages']:
    for b in page['blocks']:
        blocks[b['id']] = (page['index'], b)

issues = qaj.get('issues') or qaj.get('blockers') or []
print('QA keys:', list(qaj.keys()))
print('issue count:', len(issues))
cnt = collections.Counter()
for it in issues:
    cnt[it.get('kind') or it.get('code', '?')[:12]] += 1
print(cnt)

want = sys.argv[1] if len(sys.argv) > 1 else ''
for it in issues:
    msg = it.get('message', '')
    kind = it.get('kind', '')
    bid = it.get('block') or it.get('block_id') or ''
    if want and want not in kind:
        continue
    pg, b = blocks.get(bid, (None, None))
    line = [f"p{pg:04d}" if pg else '?', bid, kind]
    if b:
        line.append('bbox=' + ','.join(f'{v:.1f}' for v in b['bbox']))
        line.append('kind=' + str(b.get('kind')) + ' role=' + str(b.get('role')))
        line.append('act=' + str(b.get('action')) + ' ' + str(b.get('keep_kind') or ''))
        src = str(b.get('source') or b.get('text') or '')[:40].replace('\n', ' ')
        tr = str(b.get('translation') or '')[:40].replace('\n', ' ')
        line.append(f'src="{src}" tr="{tr}"')
    line.append('msg=' + msg[:150])
    print(' | '.join(line))

# figures inventory
print('\nFIGURES:')
for page in doc['pages']:
    for b in page['blocks']:
        if b.get('kind') == 'image' or b.get('role') == 'figure':
            print(f"p{page['index']:04d} {b['id']} bbox=" + ','.join(f'{v:.1f}' for v in b['bbox']), 'kind=', b.get('kind'), 'role=', b.get('role'))
