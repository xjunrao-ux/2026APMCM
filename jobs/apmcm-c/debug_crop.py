import sys
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import fitz
import pdf_translate as E
import content as C

LOG = open(r'X:\2026APMCM\jobs\apmcm-c\croplog.txt', 'w', encoding='utf-8')
orig = C.native_crop


def wrapped(source, obj):
    try:
        return orig(source, obj)
    except Exception:
        page = source[obj['page'] - 1]
        box = fitz.Rect(obj['bbox'])
        size = obj['source_size']
        lo, hi = obj.get('baseline_window', [obj['baseline'] - .95 * size, obj['baseline'] + .45 * size])

        def selected(c):
            r = fitz.Rect(c['bbox'])
            return box.contains(fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)) and lo <= c['origin'][1] <= hi

        chars = [c for b in page.get_text('rawdict')['blocks'] if b.get('type') == 0
                 for l in b['lines'] for s in l['spans'] for c in s['chars']]
        leftover = [c for c in chars if not c['c'].isspace() and not selected(c) and fitz.Rect(c['bbox']).intersects(box)]
        LOG.write('CROP FAIL page %s bbox %s baseline %.2f size %.2f window %s\n' % (
            obj['page'], [round(v, 2) for v in obj['bbox']], obj['baseline'], size, [round(lo, 2), round(hi, 2)]))
        LOG.write('   leftover: %r\n' % ([(c['c'], round(c['origin'][1], 1)) for c in leftover][:24],))
        LOG.write('   inside-selected: %r\n' % ([(c['c'], round(c['origin'][1], 1)) for c in chars
                                                 if box.contains(fitz.Point((c['bbox'][0] + c['bbox'][2]) / 2,
                                                                           (c['bbox'][1] + c['bbox'][3]) / 2))
                                                 and not c['c'].isspace()][:24],))
        LOG.flush()
        raise


C.native_crop = wrapped
sys.argv = ['pdf_translate.py', 'build', r'X:\2026APMCM\jobs\apmcm-c']
try:
    rc = E.main(sys.argv[1:])
finally:
    LOG.close()
raise SystemExit(rc)
