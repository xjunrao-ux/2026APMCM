import fitz, re, json

pdf = r'X:\2026APMCM\jobs\apmcm-c\output\translated.paired.pdf'
d = fitz.open(pdf)
word = re.compile(r'[A-Za-z][A-Za-z\-]{2,}')
out = []
for i, pg in enumerate(d, 1):
    txt = pg.get_text()
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        lat = word.findall(line)
        han = re.findall(r'[\u4e00-\u9fff]', line)
        if len(lat) >= 2 and len(lat) * 3 >= len(han):
            out.append((i, line))
for i, line in out:
    print(i, '|', line[:160])
print('total', len(out))
