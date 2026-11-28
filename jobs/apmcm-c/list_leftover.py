import json

J = r'X:\2026APMCM\jobs\apmcm-c'
doc = json.load(open(J + r'\document.json', encoding='utf-8'))
tr = json.load(open(J + r'\translations.json', encoding='utf-8'))
regs = json.load(open(J + r'\regions_all.json', encoding='utf-8'))['regions']
covered = {i for r in regs for i in r.get('source_ids', [])}
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}
leftover = []
for bid, e in tr.items():
    if e.get('keep_kind') == 'table_cell' and bid not in covered:
        leftover.append(bid)
print('leftover count', len(leftover))
for bid in sorted(leftover):
    b = B[bid]
    print(bid, 'p', b['page'], b.get('kind'), b.get('style_role'), round(b.get('font_size', 0), 1),
          [round(x, 1) for x in b['bbox']], repr(b['source'][:90]), '| text:', repr(e.get('text', '')[:30]))
