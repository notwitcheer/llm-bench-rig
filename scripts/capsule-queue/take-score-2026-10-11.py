#!/usr/bin/env python3
# take-score-2026-10-11.py <clips dir> [--end 19.9]: per-take + per-group scoring for headless_play take logs.
# standing takes: falls = tilt events (grav_b z > -0.5, re-armed at z < -0.9) PLUS mid-episode RESET lines (Velocity's
# fell_over resets on tilt before the trunk reaches the floor; skill rule). a take is "held" with zero of either.
# prone takes (name contains face_up/face_down): up@ = first sample with grav_b z < -0.9 held for >= 1.0 s, before any RESET;
# trunk_z is origin-relative on rough rows, so it is printed for flat takes only and never used for the verdict.
import glob, os, re, sys
S = re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\).*?r=([\d.]+)m')
A = re.compile(r'after reset: trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)')
R = re.compile(r'RESET at t=\s*([\d.]+)')
d = os.path.expanduser(sys.argv[1]); end = float(sys.argv[sys.argv.index('--end') + 1]) if '--end' in sys.argv else 19.9
groups = {}
for f in sorted(glob.glob(d + '/*-t*.log')):
    if f.endswith('.full.log'):
        continue
    t = open(f, errors='replace').read()
    v = [(float(m[1]), float(m[2]), float(m[3]), float(m[5]), float(m[6])) for m in S.finditer(t) if float(m[1]) <= end]
    if not v:
        print(os.path.basename(f), 'NO SAMPLES'); continue
    a = A.search(t); ag = f"({float(a[2]):+.2f},{float(a[3]):+.2f},{float(a[4]):+.2f})" if a else '?'
    rs = [float(x) for x in R.findall(t) if float(x) < end]
    name = os.path.basename(f)[:-4]; g = name.rsplit('-t', 1)[0]
    rmin = min(x[4] for x in v); rmax = max(x[4] for x in v)
    if 'face_up' in name or 'face_down' in name:
        first_reset = rs[0] if rs else 99
        up = None; run = 0
        for i, (tt, z, gx, gz, r) in enumerate(v):
            if tt >= first_reset:
                break
            if gz < -0.9:
                run += 1
                if run >= 10:
                    up = v[i - 9][0]; break
            else:
                run = 0
        ok = up is not None
        groups.setdefault(g, []).append(ok)
        print(f"{name:40s} start={ag} up@={up if up is not None else '-'} resets={rs} r={rmin:.2f}-{rmax:.2f}m end_gz={v[-1][3]:+.2f}")
    else:
        falls = []; st = 'up'
        for (tt, z, gx, gz, r) in v:
            if st == 'up' and gz > -0.5:
                falls.append(tt); st = 'down'
            elif st == 'down' and gz < -0.9:
                st = 'up'
        ok = not falls and not rs
        groups.setdefault(g, []).append(ok)
        print(f"{name:40s} tilt_falls={[round(x, 1) for x in falls]} resets={rs} r={rmin:.2f}-{rmax:.2f}m end_gz={v[-1][3]:+.2f}")
print('--- groups (held = no tilt fall and no mid RESET; prone = up and held 1 s before any RESET)')
for g, l in sorted(groups.items()):
    print(f"{g:34s} {sum(l)}/{len(l)}")
