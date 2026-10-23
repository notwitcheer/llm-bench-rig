#!/usr/bin/env python3
# worst case for z6 rival takes: count a rival take held if every event in it is an upright mid RESET (max grav_b z < -0.9 in the 0.5 s before)
import glob, os, re
from math import comb
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)')
R = re.compile(r'RESET at t=\s*([\d.]+)')
def f1(a, n1, b, n2):
    K=a+b; N=n1+n2; return sum(comb(n1,x)*comb(n2,K-x) for x in range(a, min(n1,K)+1))/comb(N,K)
def f2(a, n1, b, n2):
    K=a+b; N=n1+n2; p0=comb(n1,a)*comb(n2,b)
    return min(1.0, sum(comb(n1,x)*comb(n2,K-x) for x in range(max(0,K-n2), min(n1,K)+1) if comb(n1,x)*comb(n2,K-x) <= p0*(1+1e-9))/comb(N,K))
for c, riv in (('a','v2'),('b','z3'),('c','z2'),('d','z4')):
    d = os.path.expanduser(f'~/Pollen/clips/proof-z6-{c}-2026-10-17')
    cnt = {}
    for f in sorted(glob.glob(d + '/*-t*.log')):
        if f.endswith('.full.log'): continue
        t = open(f, errors='replace').read(); pol = os.path.basename(f).split('-')[0]
        v = [(float(m[1]), float(m[5])) for m in S.finditer(t) if float(m[1]) <= 19.9]
        rs = [float(x) for x in R.findall(t) if float(x) < 19.9]
        falls = 0; st = 'up'
        for tt, gz in v:
            if st == 'up' and gz > -0.5: falls += 1; st = 'down'
            elif st == 'down' and gz < -0.9: st = 'up'
        upr = [x for x in rs if (lambda w: w and max(w) < -0.9)([gz for tt, gz in v if x-0.5 <= tt <= x])]
        strict = falls == 0 and not rs
        lenient = falls == 0 and len(upr) == len(rs)
        h = cnt.setdefault(pol, [0, 0, 0]); h[0] += strict; h[1] += lenient; h[2] += 1
    z = cnt['z6']; o = cnt[riv]
    print(f'{c}: z6 strict {z[0]}/{z[2]} lenient {z[1]}; {riv} strict {o[0]}/{o[2]} lenient {o[1]}; worst case z6 strict vs {riv} lenient one-sided p={f1(z[0],z[2],o[1],o[2]):.4f} two-sided p={f2(z[0],z[2],o[1],o[2]):.4f}')
