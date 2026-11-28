import fitz
from PIL import Image

src = r'X:\2026APMCM\jobs\apmcm-c\output\bilingual.paired.pdf'
d = fitz.open(src)
out = r'X:\2026APMCM\jobs\apmcm-c'
for i in range(0, len(d), 2):
    imgs = []
    for j in (i, i + 1):
        if j >= len(d):
            break
        pg = d[j]
        clip = fitz.Rect(601.28, 0, 1202.56, 865.89)
        pix = pg.get_pixmap(clip=clip, dpi=150)
        imgs.append(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
    w = sum(im.width for im in imgs) + 12 * (len(imgs) - 1)
    h = max(im.height for im in imgs)
    sheet = Image.new('RGB', (w, h), 'white')
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 0))
        x += im.width + 12
    sheet.save(rf'{out}\zhrev_{i//2+1:02d}.png')
    print(i//2+1, imgs[0].width, h)
