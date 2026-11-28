import json

J = r'X:\2026APMCM\jobs\apmcm-c'
doc = json.load(open(J + r'\document.json', encoding='utf-8'))
tr = json.load(open(J + r'\translations.json', encoding='utf-8'))
regs = json.load(open(J + r'\regions_all.json', encoding='utf-8'))['regions']
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}


def inter(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


for r in regs:
    if r['kind'] != 'table':
        continue
    labels = r.get('labels', [])
    for bid in r.get('source_ids', []):
        b = B[bid]
        import re
        if b.get('kind') != 'text' or not re.search(r'[A-Za-z]{2,}', b['source']):
            continue
        mine = [l for l in labels if inter(l['bbox'], b['bbox']) > 0.5 * (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1])]
        if not mine:
            print(r['id'], bid, repr(b['source'][:70]))
