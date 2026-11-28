import json
from fontTools.ttLib import TTFont

data = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\regions_all.json', encoding='utf-8'))
regions = data['regions'] if isinstance(data, dict) else data
f = TTFont(r'C:\Windows\Fonts\simhei.ttf')
cmap = set()
for t in f['cmap'].tables:
    cmap |= set(t.cmap)
bad = []
for reg in regions:
    for lab in reg.get('labels') or []:
        miss = sorted({c for c in lab['text'] if ord(c) not in cmap})
        if miss:
            bad.append((reg['id'], lab['text'][:50], miss))
print('labels missing glyphs:', len(bad))
for b in bad:
    print(b[0], repr(b[1]), b[2])
