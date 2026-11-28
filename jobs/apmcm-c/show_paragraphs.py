import json

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
tr = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\translations.json', encoding='utf-8'))
ids = ['p0008-b0021', 'p0010-b0020', 'p0011-b0067', 'p0016-b0028', 'p0019-b0057', 'p0021-b0004']
for page in doc['pages']:
    for b in page['blocks']:
        if b['id'] in ids:
            e = tr.get(b['id'], {})
            print('=====', b['id'], 'page', page['number'])
            print('SOURCE:', repr(b.get('source')))
            print('TRANS:', repr(e.get('text')))
            for m in b.get('protected_math') or []:
                print('   ', m['token'], 'safe', m.get('safe'), 'h', round(m['bbox'][3] - m['bbox'][1], 1),
                      'src', repr(m.get('source')))
