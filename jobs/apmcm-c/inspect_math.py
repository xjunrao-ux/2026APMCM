import json

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
for page in doc['pages']:
    for b in page['blocks']:
        if b['id'] in ('p0010-b0020',):
            print('BLOCK', b['id'], b['bbox'], 'kind', b['kind'], 'fs', b.get('font_size'))
            print('SOURCE:', repr(b.get('source'))[:400])
            for m in b.get('protected_math') or []:
                print('  ', m['token'], [round(v, 2) for v in m['bbox']], 'baseline', round(m['baseline'], 2),
                      'size', round(m['source_size'], 2), 'safe', m.get('safe'), 'src', repr(m.get('source'))[:60])
regions = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\regions_all.json', encoding='utf-8'))
rs = regions['regions'] if isinstance(regions, dict) else regions
for r in rs:
    if r['id'].startswith('p0010-b0020'):
        print('REGION', r['id'], r['bbox'])
        for m in r.get('native_math') or []:
            print('  ', m['token'], [round(v, 2) for v in m['bbox']], 'baseline', round(m.get('baseline', -1), 2),
                  'size', round(m.get('source_size', -1), 2))
