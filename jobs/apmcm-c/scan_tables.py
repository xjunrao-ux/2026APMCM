#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cluster small-font blocks (table cells) per page with union bboxes."""
import json, os, sys

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
lo, hi = (int(x) for x in sys.argv[1].split('-')) if len(sys.argv) > 1 else (1, 99)
for page in doc['pages']:
    n = page['number']
    if not (lo <= n <= hi):
        continue
    H = page['height']
    small = [b for b in page['blocks']
             if b.get('font_size', 99) <= 9.6 and b['bbox'][1] > 60 and b['bbox'][3] < H - 45
             and b.get('kind') not in ('image',)]
    if not small:
        continue
    small.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
    clusters = []
    for b in small:
        y0, y1 = b['bbox'][1], b['bbox'][3]
        if clusters and y0 - clusters[-1]['y1'] <= 6:
            c = clusters[-1]
            c['y1'] = max(c['y1'], y1); c['x0'] = min(c['x0'], b['bbox'][0]); c['x1'] = max(c['x1'], b['bbox'][2])
            c['ids'].append(b['id'])
        else:
            clusters.append(dict(y0=y0, y1=y1, x0=b['bbox'][0], x1=b['bbox'][2], ids=[b['id']]))
    print(f"--- page {n} h={H:.0f}")
    for c in clusters:
        first = next(b for b in small if b['id'] == c['ids'][0])
        last = next(b for b in small if b['id'] == c['ids'][-1])
        print(f"  y {c['y0']:6.1f}-{c['y1']:6.1f} x {c['x0']:6.1f}-{c['x1']:6.1f} n={len(c['ids']):3d} {c['ids'][0]}..{c['ids'][-1]} | {str(first.get('source'))[:40]} .. {str(last.get('source'))[:40]}")
