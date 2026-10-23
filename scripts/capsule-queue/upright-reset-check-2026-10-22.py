#!/usr/bin/env python3
# upright-reset check for proof-z5/z6 (2026-10-22): a mid RESET with max grav_b z < -0.9 in the 0.5 s before = truncation with the duck upright
import glob, os, re
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)')
R = re.compile(r'RESET at t=\s*([\d.]+)')
for d in sorted(glob.glob(os.path.expanduser('~/Pollen/clips/proof-z[56]-?-2026-10-1?'))):
    up = {}; tot = {}; nb = 0; nlog = 0
    for f in sorted(glob.glob(d + '/*-t*.log')):
        if f.endswith('.full.log'): continue
        nlog += 1
        full = f[:-4] + '.full.log'
        if os.path.exists(full) and 'out_of_terrain_bounds dropped' in open(full, errors='replace').read(): nb += 1
        t = open(f, errors='replace').read()
        v = [(float(m[1]), float(m[5])) for m in S.finditer(t)]
        pol = os.path.basename(f).split('-')[0]
        for x in R.findall(t):
            x = float(x)
            if x >= 19.9: continue
            tot[pol] = tot.get(pol, 0) + 1
            w = [gz for tt, gz in v if x - 0.5 <= tt <= x]
            if w and max(w) < -0.9:
                up.setdefault(pol, []).append((os.path.basename(f)[:-4], x))
    print(os.path.basename(d), f'logs={nlog} no-bounds-line={nb}', 'mid resets', tot, 'upright', up)
