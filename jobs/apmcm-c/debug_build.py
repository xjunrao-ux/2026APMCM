import sys, json
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import pdf_translate as E
import fitz

LOG = open(r'X:\2026APMCM\jobs\apmcm-c\debuglog.txt', 'w', encoding='utf-8')
orig = E.typography.rich_layout


def debug(text, styles, rect, block, font, min_size, wrap):
    plan = orig(text, styles, rect, block, font, min_size, wrap)
    if plan is None:
        LOG.write('FAIL %r rect %s size %s font %r w %.2f h %.2f\n' % (
            text, [round(v, 2) for v in rect], min_size,
            getattr(font, 'name', '?'), rect.width, rect.height))
        try:
            lines = wrap(text, rect.width - .04 * min_size, font, min_size)
            LOG.write('   wrap -> %r\n' % (lines,))
        except Exception as exc:
            LOG.write('   wrap error %s\n' % exc)
    return plan


E.typography.rich_layout = debug
sys.argv = ['pdf_translate.py', 'build', r'X:\2026APMCM\jobs\apmcm-c']
try:
    rc = E.main(sys.argv[1:])
finally:
    LOG.close()
raise SystemExit(rc)
