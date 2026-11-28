import json, re, sys
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import fitz
from collections import Counter
NUMBER = re.compile(r"(?<![\w])[-+\u2212]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
regions = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\regions_all.json', encoding='utf-8'))
regions = regions['regions'] if isinstance(regions, dict) else regions
src = fitz.open(r'X:\2026APMCM\jobs\apmcm-c\source.pdf')
for reg in regions:
    if reg['kind'] != 'paragraph':
        continue
    page = src[reg['page'] - 1]
    r = fitz.Rect(reg['bbox'])
    raw = page.get_text('rawdict', clip=r)
    lines = []
    for b in raw['blocks']:
        for line in b.get('lines', []):
            chars = [c for s in line['spans'] for c in s['chars']]
            lines.append(''.join(c['c'] for c in chars
                                 if r.contains(fitz.Point((c['bbox'][0] + c['bbox'][2]) / 2,
                                                          (c['bbox'][1] + c['bbox'][3]) / 2))))
    original = '\n'.join(lines)
    a = Counter(NUMBER.findall(original))
    b = Counter(NUMBER.findall(reg['translation']))
    print('=====', reg['id'])
    print('src', dict(a))
    print('zh ', dict(b))
    print('diff', dict(a - b), dict(b - a))
