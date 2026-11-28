import json

qa = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\output\qa.json', encoding='utf-8'))
doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
tr = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\translations.json', encoding='utf-8'))
B = {b['id']: b for p in doc['pages'] for b in p['blocks']}
for q in qa['blockers']:
    bid = q.get('block_id')
    print('=====', q['id'], q.get('message'), q.get('detail'))
    if bid:
        b = B[bid]
        print('SOURCE:', repr(b.get('source')))
        print('TRANS :', repr(tr.get(bid, {}).get('text')))
        print('math  :', [(m['token'], m.get('source')) for m in b.get('protected_math') or []])
