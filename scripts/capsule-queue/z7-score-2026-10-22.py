#!/usr/bin/env python3
# z7-score-2026-10-22.py: 09z7 (mixslope) verdict scoring. held = no tilt fall and no mid RESET (take-score-2026-10-11 rule),
# exact Fisher via math.comb. pre-registered: a = gain condition vs walk-mix (one-sided); b (vs walk-slope) and c (vs walk-mix on
# its own condition) = no-loss checks (two-sided; a loss = significant two-sided result with z7 lower).
# verdict line: GAIN if a one-sided p < 0.05 and no loss on b/c; NO GAIN otherwise (loss named when present).
import glob, os, re
from math import comb
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\).*?r=([\d.]+)m')
R = re.compile(r'RESET at t=\s*([\d.]+)')
END = 19.9
def score(d):
    out = {}
    for f in sorted(glob.glob(d + '/*-t*.log')):
        if f.endswith('.full.log'): continue
        t = open(f, errors='replace').read()
        v = [(float(m[1]), float(m[5])) for m in S.finditer(t) if float(m[1]) <= END]
        pol = os.path.basename(f).split('-')[0]
        if not v: out.setdefault(pol, []).append(None); continue
        rs = [float(x) for x in R.findall(t) if float(x) < END]
        falls = 0; st = 'up'
        for tt, gz in v:
            if st == 'up' and gz > -0.5: falls += 1; st = 'down'
            elif st == 'down' and gz < -0.9: st = 'up'
        out.setdefault(pol, []).append(falls == 0 and not rs)
    return out
def f1(a, n1, b, n2):
    K = a + b; N = n1 + n2
    return sum(comb(n1, x) * comb(n2, K - x) for x in range(a, min(n1, K) + 1)) / comb(N, K)
def f2(a, n1, b, n2):
    K = a + b; N = n1 + n2; p0 = comb(n1, a) * comb(n2, b)
    return min(1.0, sum(comb(n1, x) * comb(n2, K - x) for x in range(max(0, K - n2), min(n1, K) + 1) if comb(n1, x) * comb(n2, K - x) <= p0 * (1 + 1e-9)) / comb(N, K))
res = {}
for c, me, rival, label in (('a', 'z7a', 'z6a', 'walk-mix'), ('b', 'z7b', 'z5b', 'walk-slope'), ('c', 'z7c', 'z6c', 'walk-mix')):
    r = score(os.path.expanduser(f'~/Pollen/clips/proof-z7-{c}-2026-10-22'))
    z = [x for x in r.get(me, []) if x is not None]; o = [x for x in r.get(rival, []) if x is not None]
    a, n1, b, n2 = sum(z), len(z), sum(o), len(o)
    if not n1 or not n2: print(f'{c}: no takes yet'); res[c] = None; continue
    p1, p2 = f1(a, n1, b, n2), f2(a, n1, b, n2); res[c] = (a, n1, b, n2, p1, p2)
    print(f'{c}: z7 {a}/{n1} vs {label} {b}/{n2}  one-sided p={p1:.4f}  two-sided p={p2:.4f}  no-sample takes z7 {r.get(me,[]).count(None)} {rival} {r.get(rival,[]).count(None)}')
if all(res.get(c) and res[c][1] == 15 and res[c][3] == 15 for c in 'abc'):
    gain = res['a'][4] < 0.05
    loss = [c for c in 'bc' if res[c][5] < 0.05 and res[c][0] / res[c][1] < res[c][2] / res[c][3]]
    print('VERDICT:', ('GAIN' if gain and not loss else 'NO GAIN') + (f' (loss on {",".join(loss)})' if loss else ''))
else:
    print('VERDICT: incomplete (not every condition has 15 + 15 takes yet)')
