import json, glob, os, re
from collections import Counter

JOB = r'X:\2026APMCM\jobs\apmcm-c'
key = 'd79a7e171f16855efb27975e44aeb149999931f1ce308cd747d70be01fe687bb'
pack = json.load(open(os.path.join(JOB, 'packets', key + '.json'), encoding='utf-8'))
snap = pack['snapshots']  # id -> input_hash; entries: id/text
entries = {e[0]: e[1] for e in pack['entries']}
TAG = re.compile(r'</?s\d+>')
PH = re.compile(r'⟦[MT]\d+⟧')
NUMBER = re.compile(r"(?<![\w])[-+−]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")

def nums(s):
    return Counter(NUMBER.findall(s))

errors = []
numrep = []
for fn in sorted(glob.glob(os.path.join(JOB, 'replies', 'slice_*.json')) + glob.glob(os.path.join(JOB, 'replies', 'lead_*.json'))):
    d = json.load(open(fn, encoding='utf-8'))
    name = os.path.basename(fn)
    if d.get('packet') != key:
        errors.append((name, 'packet mismatch'))
    ids = [it[0] for it in d['items']]
    if len(set(ids)) != len(ids):
        errors.append((name, 'dup ids'))
    for it in d['items']:
        bid = it[0]
        if bid not in entries:
            errors.append((name, 'unknown id ' + bid)); continue
        src = entries[bid]
        text = it[1] if len(it) > 1 else ''
        if len(it) > 2 and it[2].get('keep'):
            continue
        if TAG.findall(src) != TAG.findall(text):
            errors.append((name, 'tag mismatch ' + bid + ' ' + str(TAG.findall(src)) + ' vs ' + str(TAG.findall(text))))
        if PH.findall(src) != PH.findall(text):
            errors.append((name, 'PH mismatch ' + bid + ' ' + str(PH.findall(src)) + ' vs ' + str(PH.findall(text))))
        if nums(src) != nums(text):
            numrep.append((name, bid, dict(nums(src)), dict(nums(text)), src[:70], text[:70]))
print('ERRORS', len(errors))
for e in errors[:40]:
    print(e)
print('NUM MISMATCH', len(numrep))
for r in numrep:
    print(r[0], r[1], 'src', r[2], 'dst', r[3])
    print('   S:', r[4])
    print('   T:', r[5])
