import json, re
import fitz

J = r'X:\2026APMCM\jobs\apmcm-c'
doc = json.load(open(J + r'\document.json', encoding='utf-8'))
tr = json.load(open(J + r'\translations.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}

FD = fitz.Font(fontfile=r'C:\Windows\Fonts\simsun.ttc')


def w(s):
    return FD.text_length(s, 10.5)


ids = []
for idx, p in enumerate(doc['pages'], 1):
    if idx in (2, 3):
        for b in p['blocks']:
            if b['id'] in tr and re.search(r'(\s*\.\s*){5,}', b['source']) and b['kind'] == 'text':
                ids.append(b['id'])

items = []
for bid in ids:
    b = B[bid]
    text = tr[bid]['text']
    prefix = re.sub(r'(\s*\.\s*)+$', '', text).rstrip()
    avail = 480.0 - b['bbox'][0] - 3.5
    pw = w(prefix)
    k = 1
    while k < 400:
        s = prefix + '  ' + ' '.join(['.'] * k)
        if w(s) > avail:
            break
        k += 1
    k = max(1, k - 1)
    new = prefix + '  ' + ' '.join(['.'] * k)
    if w(new) > avail:
        new = prefix
    items.append([bid, new])
    print(bid, round(w(prefix), 1), round(avail, 1), k, round(w(new), 1))

reply = dict(packet='d79a7e171f16855efb27975e44aeb149999931f1ce308cd747d70be01fe687bb', items=items)
json.dump(reply, open(J + r'\replies\toc_fix2.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('items', len(items))
