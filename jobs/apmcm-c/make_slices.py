import json, os, glob
from collections import OrderedDict

JOB = r'X:\2026APMCM\jobs\apmcm-c'
PAT = os.path.join(JOB, 'packets', '*.json')
packets = sorted(glob.glob(PAT), key=os.path.getmtime)
p = packets[-1]
d = json.load(open(p, encoding='utf-8'))
key = os.path.basename(p)[:-5]
items = d['entries']
pg = OrderedDict()
for it in items:
    pg.setdefault(it[0].split('-')[0], []).append(it)
pages = list(pg)

def emit(name, pages_sel):
    arr = []
    for x in pages_sel:
        arr.extend(pg[x])
    out = {'packet': key, 'pages': pages_sel, 'items': arr}
    fn = os.path.join(JOB, 'slices', name)
    json.dump(out, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(name, pages_sel[0] + '-' + pages_sel[-1], len(arr), sum(len(i[1]) for i in arr))

os.makedirs(os.path.join(JOB, 'slices'), exist_ok=True)
emit('lead_p01_p02.json', ['p0001', 'p0002'])
rest = pages[2:]
groups = []
cur = []
c = 0
for x in rest:
    cur.append(x)
    c += sum(len(i[1]) for i in pg[x])
    if c >= 6000:
        groups.append(cur)
        cur = []
        c = 0
if cur:
    groups.append(cur)
for i, g in enumerate(groups, 1):
    emit('slice_%02d.json' % i, g)
print('packet', key)
