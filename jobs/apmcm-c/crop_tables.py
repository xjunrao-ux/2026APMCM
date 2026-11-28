import fitz, json
from PIL import Image

J = r'X:\2026APMCM\jobs\apmcm-c'
regs = json.load(open(J + r'\regions_all.json', encoding='utf-8'))['regions']
tables = [r for r in regs if r['kind'] == 'table']
print('tables', len(tables))
d = fitz.open(J + r'\output\translated.paired.pdf')
crops = []
for r in tables:
    pg = d[r['page'] - 1]
    x0, y0, x1, y1 = r['bbox']
    clip = fitz.Rect(max(0, x0 - 3), max(0, y0 - 3), min(pg.rect.width, x1 + 3), min(pg.rect.height, y1 + 3))
    pix = pg.get_pixmap(clip=clip, dpi=200)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    crops.append((r['id'], im))
# compose sheets: 2 per sheet stacked vertically
for i in range(0, len(crops), 2):
    group = crops[i:i + 2]
    w = max(im.width for _, im in group)
    h = sum(im.height for _, im in group) + 10 * (len(group) - 1)
    sheet = Image.new('RGB', (w, h), 'white')
    y = 0
    for _, im in group:
        sheet.paste(im, (0, y))
        y += im.height + 10
    path = rf'{J}\tbl_{i//2+1:02d}.png'
    sheet.save(path)
    print(path, [g[0] for g in group], sheet.size)
