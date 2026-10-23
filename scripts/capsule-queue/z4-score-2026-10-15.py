#!/usr/bin/env python3
# z4-score-2026-10-15.py: Fisher p per condition from take-score groups + mean body tilt in held takes (09z4 verdict).
import glob, math, os, re, subprocess, sys
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)')
R = re.compile(r'RESET at t=\s*([\d.]+)')
base = os.path.expanduser('~/Pollen/clips/proof-z4-%s-2026-10-15')
def held(f, end=19.9):
    t = open(f, errors='replace').read()
    v = [(float(m[1]), float(m[5])) for m in S.finditer(t) if float(m[1]) <= end]
    rs = [float(x) for x in R.findall(t) if float(x) < end]
    falls = []; st = 'up'
    for tt, gz in v:
        if st == 'up' and gz > -0.5: falls.append(tt); st = 'down'
        elif st == 'down' and gz < -0.9: st = 'up'
    tilts = [math.degrees(math.acos(max(-1.0, min(1.0, -gz)))) for tt, gz in v if tt >= 1.0]
    return (bool(v) and not falls and not rs), tilts, bool(v)
def fisher(a, n1, b, n2):  # P(X >= a) for policy1 holds under hypergeometric; and two-sided
    K = a + b; N = n1 + n2
    p = lambda k: math.comb(n1, k) * math.comb(n2, K - k) / math.comb(N, K)
    ks = range(max(0, K - n2), min(n1, K) + 1)
    one = sum(p(k) for k in ks if k >= a)
    p0 = p(a); two = sum(p(k) for k in ks if p(k) <= p0 * (1 + 1e-9))
    return one, min(1.0, two)
for c in 'abc':
    d = base % c; res = {}
    for pol in ('z4', 'v2'):
        fs = [f for f in sorted(glob.glob(f'{d}/{pol}-{c}-t*.log')) if not f.endswith('.full.log')]
        rows = [held(f) for f in fs]
        nsamp = sum(1 for r in rows if r[2])
        h = sum(1 for r in rows if r[0])
        ht = [x for r in rows if r[0] for x in r[1]]
        res[pol] = (h, nsamp, len(fs), (sum(ht) / len(ht)) if ht else float('nan'), (sorted(ht)[int(0.95 * len(ht))] if ht else float('nan')))
    one, two = fisher(res['z4'][0], res['z4'][1], res['v2'][0], res['v2'][1])
    mp4 = len(glob.glob(f'{d}/*.mp4'))
    print(f"{c}: z4 {res['z4'][0]}/{res['z4'][1]} v2 {res['v2'][0]}/{res['v2'][1]}  one-sided(z4>v2) p={one:.4f} two-sided p={two:.4f}  logs z4 {res['z4'][2]} v2 {res['v2'][2]} mp4 {mp4}  held-take tilt mean z4 {res['z4'][3]:.1f} v2 {res['v2'][3]:.1f} deg, p95 z4 {res['z4'][4]:.1f} v2 {res['v2'][4]:.1f}")
