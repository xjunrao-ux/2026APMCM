import sys, json
from fontTools.ttLib import TTFont

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
tr = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\translations.json', encoding='utf-8'))
f = TTFont(r'C:\Windows\Fonts\simhei.ttf')
cmap = set()
for t in f['cmap'].tables:
    cmap |= set(t.cmap)
for bid in ['p0023-b0018', 'p0023-b0019']:
    txt = tr[bid]['text']
    miss = sorted({c for c in txt if not c.isspace() and ord(c) not in cmap})
    print(bid, [hex(ord(c)) + ' ' + c for c in miss], repr(txt[:80]))
