import json, glob, os, re
from collections import Counter

JOB = r'X:\2026APMCM\jobs\apmcm-c'
key = 'd79a7e171f16855efb27975e44aeb149999931f1ce308cd747d70be01fe687bb'
pack = json.load(open(os.path.join(JOB, 'packets', key + '.json'), encoding='utf-8'))
entries = {e[0]: e[1] for e in pack['entries']}
NUMBER = re.compile(r"(?<![\w])[-+−]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
NUM_PR = re.compile(r'(?<![\d.])(\d+)\s+\.\s+(\d+)(?![\d.])')

OVR = {
 'p0001-b0007': None,  # handled by word-number fix below
}

WORD_NUM = {
 'p0001-b0007': [('7 个距离', '七个距离')],
 'p0014-b0032': [('74 个实验', '七十四个实验'), ('2 个仅', '两个仅')],
 'p0017-b0009': [('30% 的接近', '百分之三十的接近')],
}

files = sorted(glob.glob(os.path.join(JOB, 'replies', '*.json')))
changed = 0
for fn in files:
    d = json.load(open(fn, encoding='utf-8'))
    mod = False
    for it in d['items']:
        bid, text = it[0], it[1]
        src = entries.get(bid, '')
        if Counter(NUMBER.findall(src)) == Counter(NUMBER.findall(text)):
            continue
        # 1) restore spaced section numbering "N . M"
        m = NUM_PR.search(src)
        if m:
            a, b = m.group(1), m.group(2)
            pat = a + '.' + b
            if pat in text:
                text = text.replace(pat, a + ' . ' + b, 1)
        # 2) word-number overrides
        for old, new in WORD_NUM.get(bid, []):
            text = text.replace(old, new)
        it[1] = text
        if Counter(NUMBER.findall(src)) != Counter(NUMBER.findall(text)):
            print('STILL', os.path.basename(fn), bid, dict(Counter(NUMBER.findall(src))), dict(Counter(NUMBER.findall(text))))
        else:
            changed += 1
        mod = True
    if mod:
        json.dump(d, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('fixed items', changed)
