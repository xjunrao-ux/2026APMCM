import sys, argparse
sys.path.insert(0, r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts')
import fitz
import pdf_translate as E
import reviewed_regions as R

job = r'X:\2026APMCM\jobs\apmcm-c'
doc = E.read_json(job + r'\document.json')
args = E.parser().parse_args(['build', job])
ns = argparse.Namespace(**vars(E))
resolver = ns.font_policy.Resolver(args)

n = 0
for group in doc.get('reviewed_regions', []):
    for label in group.get('labels', []):
        n += 1
        box = fitz.Rect(label['bbox'])
        font, record = resolver.for_block(dict(id=group['id'] + '-label', spans=[dict(font='Helvetica', flags=0)]))
        size = label.get('font_size', 7.5)
        rot = label.get('rotation', 0)
        missing = [c for c in label['text'] if not c.isspace() and not font.has_glyph(ord(c))]
        if missing:
            print('GLYPH', group['id'], repr(label['text'][:40]), missing)
            continue
        b = dict(font_size=size, spans=[dict(font='Helvetica', flags=0, text=label['text'], size=size)],
                 terms=[], align=label.get('align', 'center'))
        text, styles = ns.typography.styled_text(b, dict(text=label['text']))
        rect = fitz.Rect(3, 3, 3 + box.height, 3 + box.width) if rot == 90 else box
        plan = ns.typography.rich_layout(text, styles, rect, b, font, size, ns.wrap_text)
        if plan is None:
            try:
                lines = ns.wrap_text(text, rect.width - .04 * size, font, size)
            except Exception as exc:
                lines = 'wrap error %s' % exc
            print('FIT', group['id'], repr(label['text']), 'w %.2f h %.2f size %s' % (rect.width, rect.height, size), lines)
print('labels checked', n)
