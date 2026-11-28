import json, re
from collections import Counter

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
tr = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\translations.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}
NUMBER = re.compile(r"(?<![\w])[-+\u2212]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
TOKEN = re.compile(r'⟦M\d{4}⟧')
for bid in ['p0011-b0003', 'p0014-b0035', 'p0017-b0012']:
    b = B[bid]
    src = b['source']
    txt = tr[bid]['text']
    msrc = [m['source'] for m in b.get('protected_math') or []]
    print('=====', bid)
    print('src  all', dict(Counter(NUMBER.findall(src))))
    print('src math-only', dict(Counter(NUMBER.findall(''.join(msrc)))))
    print('zh   all', dict(Counter(NUMBER.findall(TOKEN.sub('', txt)))))
