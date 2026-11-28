import sys, json, re
from collections import Counter
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import pdf_translate as E

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
tr = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\translations.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}
for bid in ['p0011-b0003', 'p0014-b0035', 'p0017-b0012']:
    b = B[bid]
    entry = tr[bid]
    exp = E.expanded(b, entry, numbers=True)
    a = Counter(E.NUMBER.findall(b['source']))
    c = Counter(E.NUMBER.findall(exp))
    print('=====', bid)
    print('expanded:', exp[-400:])
    print('src ', dict(a))
    print('exp ', dict(c))
    print('src-exp', dict(a - c), 'exp-src', dict(c - a))
