import sys, argparse, json, re
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import fitz
import pdf_translate as E

job = r'X:\2026APMCM\jobs\apmcm-c'
doc = E.read_json(job + r'\document.json')
tr = E.read_json(job + r'\translations.json')
args = E.parser().parse_args(['build', job])
ns = argparse.Namespace(**vars(E))
resolver = ns.font_policy.Resolver(args)
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}

ids = []
for idx, p in enumerate(doc['pages'], 1):
    if idx in (2, 3):
        for b in p['blocks']:
            if b['id'] in tr and re.search(r'(\s*\.\s*){5,}', b['source']) and b['kind'] == 'text':
                ids.append(b['id'])

items = []
for bid in ids:
    b = B[bid]
    font, _ = resolver.for_block(b)
    size = 12.5 if b.get('role') == 'heading' else b['font_size']
    limit = fitz.Rect(b['bbox']).width - 6.0
    text = tr[bid]['text']
    prefix = re.sub(r'(\s*\.\s*)+$', '', text).rstrip()
    best = prefix
    k = 1
    while k < 200:
        cand = prefix + '  ' + ' '.join(['.'] * k)
        if font.text_length(cand, size) > limit:
            break
        best = cand
        k += 1
    items.append([bid, best])
    print(bid, round(font.text_length(prefix, size), 1), round(limit, 1), k - 1, round(font.text_length(best, size), 1))

json.dump(dict(packet='d79a7e171f16855efb27975e44aeb149999931f1ce308cd747d70be01fe687bb', items=items),
          open(job + r'\replies\toc_fix3.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('items', len(items))
