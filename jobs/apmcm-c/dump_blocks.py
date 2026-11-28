#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dump blocks of selected pages: id, bbox, kind, action, font size, source."""
import json, os, sys

JOB = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(JOB, 'document.json'), encoding='utf-8'))
pages = sorted(int(x) for x in sys.argv[1].split(','))
for page in doc['pages']:
    if page['number'] not in pages:
        continue
    print(f"===== PAGE {page['number']} =====")
    for b in page['blocks']:
        pg_bbox = ','.join(f'{v:.1f}' for v in b['bbox'])
        src = str(b.get('source') or '').replace('\n', '\\n')
        print(f"{b['id']:12} {pg_bbox:32} {b.get('kind','')[:6]:6} fs={b.get('font_size',0):4.1f} rot={b.get('rotation',0)} act={b.get('action')} {src[:90]}")
