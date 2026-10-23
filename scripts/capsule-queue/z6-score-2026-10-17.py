#!/usr/bin/env python3
# z6-score-2026-10-17.py: 09z6 (mix) verdict scoring. held = no tilt fall and no mid RESET (take-score-2026-10-11 rule),
# per condition z6 vs its rival, exact Fisher via math.comb. a = pre-registered gain condition (vs walk-flat v2, one-sided);
# b/c/d = no-loss checks vs the single-condition specialist (two-sided; a loss is a significant two-sided result against z6).
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
for c, rival in (('a', 'v2'), ('b', 'z3'), ('c', 'z2'), ('d', 'z4')):
    r = score(os.path.expanduser(f'~/Pollen/clips/proof-z6-{c}-2026-10-17'))
    z = [x for x in r.get('z6', []) if x is not None]; o = [x for x in r.get(rival, []) if x is not None]
    a, n1, b, n2 = sum(z), len(z), sum(o), len(o)
    if not n1 or not n2: print(f'{c}: no takes yet'); continue
    print(f'{c}: z6 {a}/{n1} vs {rival} {b}/{n2}  one-sided p={f1(a, n1, b, n2):.4f}  two-sided p={f2(a, n1, b, n2):.4f}  no-sample takes z6 {r.get("z6",[]).count(None)} {rival} {r.get(rival,[]).count(None)}')
