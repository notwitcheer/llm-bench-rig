#!/usr/bin/env python3
# z5-score-2026-10-16.py: 09z5 verdict scoring (copy of z3-score-2026-10-14.py). held = no tilt fall and no mid RESET (same rule
# as take-score-2026-10-11.py), falls located by r= in the 0.5 s before each tilt fall / mid RESET: < 0.95 m = top (the flat hill
# top, its edge is 1.0-1.15 m from the tile centre), 0.95-3.0 m = slope (down the hill, the valley at the tile edge 2.0-2.8 m, up
# the next hill), else other. at 0.001 (control c) the bands are distance bands only, no real slope.
import glob, os, re, sys
from math import comb
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\).*?r=([\d.]+)m')
R = re.compile(r'RESET at t=\s*([\d.]+)')
END = 19.9
def score(d):
    out = {}
    for f in sorted(glob.glob(d + '/*-t*.log')):
        if f.endswith('.full.log'):
            continue
        t = open(f, errors='replace').read()
        v = [(float(m[1]), float(m[5]), float(m[6])) for m in S.finditer(t) if float(m[1]) <= END]
        name = os.path.basename(f)[:-4]; pol = name.split('-')[0]
        if not v:
            out.setdefault(pol, []).append((name, None, [])); continue
        rs = [float(x) for x in R.findall(t) if float(x) < END]
        falls = []; st = 'up'
        for (tt, gz, r) in v:
            if st == 'up' and gz > -0.5:
                falls.append(tt); st = 'down'
            elif st == 'down' and gz < -0.9:
                st = 'up'
        # events: tilt falls, plus resets not within 1 s after a tilt fall
        ev = [('tilt', x) for x in falls] + [('reset', x) for x in rs if not any(0 <= x - y <= 1.0 for y in falls)]
        locs = []
        for kind, te in sorted(ev, key=lambda e: e[1]):
            w = [r for (tt, gz, r) in v if te - 0.5 <= tt <= te]
            if not w:
                w = [min(v, key=lambda s: abs(s[0] - te))[2]]
            rr = max(w)
            loc = 'slope' if 0.95 <= rr <= 3.0 else ('top' if rr < 0.95 else 'other')
            locs.append((kind, te, round(rr, 2), loc))
        held = not falls and not rs
        out.setdefault(pol, []).append((name, held, locs))
    return out
def fisher_one(a, n1, b, n2):  # P(X >= a) for group1 successes under hypergeometric
    K = a + b; N = n1 + n2
    tot = comb(N, K)
    return sum(comb(n1, x) * comb(n2, K - x) for x in range(a, min(n1, K) + 1)) / tot
def fisher_two(a, n1, b, n2):
    K = a + b; N = n1 + n2; tot = comb(N, K)
    p0 = comb(n1, a) * comb(n2, b)
    return min(1.0, sum(comb(n1, x) * comb(n2, K - x) for x in range(max(0, K - n2), min(n1, K) + 1)
               if comb(n1, x) * comb(n2, K - x) <= p0 * (1 + 1e-9)) / tot)
res = {}
for c in 'abc':
    d = os.path.expanduser(f'~/Pollen/clips/proof-z5-{c}-2026-10-16')
    res[c] = score(d)
    print(f'=== condition {c}')
    for pol in ('z5', 'v2'):
        l = res[c].get(pol, [])
        for name, held, locs in l:
            print(f'  {name:12s} held={held} {locs}')
        h = sum(1 for x in l if x[1]); n = len(l)
        firsts = [x[2][0][3] for x in l if x[2]]
        slope_takes = sum(1 for x in l if any(e[3] == 'slope' for e in x[2]))
        print(f'  {pol}: held {h}/{n}; first-event location {dict((k, firsts.count(k)) for k in set(firsts))}; takes with a slope-band event {slope_takes}')
def hn(cs, pol):
    l = [x for c in cs for x in res[c].get(pol, [])]
    return sum(1 for x in l if x[1]), len(l)
for label, cs in (('a', 'a'), ('b', 'b'), ('a+b', 'ab'), ('c', 'c')):
    a, n1 = hn(cs, 'z5'); b, n2 = hn(cs, 'v2')
    print(f'{label}: z5 {a}/{n1} vs v2 {b}/{n2}  one-sided p={fisher_one(a, n1, b, n2):.4f}  two-sided p={fisher_two(a, n1, b, n2):.4f}')
