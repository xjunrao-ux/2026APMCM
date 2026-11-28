import sys, json, traceback
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import pdf_translate as E
import audit

job = r'X:\2026APMCM\jobs\apmcm-c'
try:
    E.main(['build', job])
except Exception as ex:
    print('FAIL:', ex)
    d, doc, tr = E.state(job)
    qa = E.read_json(E.Path(job) / 'output' / 'qa.json')
    b = next(b for b in E.blocks(doc) if b['id'] == 'p0020-b0002')
    print('block', b['id'], b['bbox'], repr(b['source']), tr.get('p0020-b0002'))
    pairs = [p for p in qa.get('paragraph_pairs', []) if 'p0020-b0002' in p['block_ids']]
    print('pairs', json.dumps(pairs, ensure_ascii=False)[:800])
    # neighbouring pairs on page 20
    for p in pairs:
        print('PAIR page', p['page'], 'kind', p['kind'], 'source_rect', p['source_rect'], 'top', p['top'])
        import fitz
        mono = fitz.open(qa['translated'])
        r = fitz.Rect(p['source_rect'])
        r += (p['mono_x'] - r.x0, p['top'] - r.y0, p['mono_x'] - r.x0, p['top'] - r.y0)
        print('mapped rect', r, 'text=', repr(mono[p['page'] - 1].get_text(clip=r + (-.75, -.75, .75, .75))))
