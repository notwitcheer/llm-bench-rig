#!/usr/bin/env python3
# recov-gz-2026-10-11.py <dir> [--end S]: carpet-recov.py for ROUGH rows: trunk_z is origin-relative there, so states come from
# body-frame gravity only. down = tilt > 60 deg; up again = grav_b z < -0.9. per take: down states (incl. a ground spawn), rises,
# rise times, upright fraction, RESET lines inside the window. prints per-policy+spawn totals at the end.
import glob, math, os, re, sys
pat = re.compile(r"t=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*(-?[\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)")
d = sys.argv[1]; end = float(sys.argv[sys.argv.index('--end') + 1]) if '--end' in sys.argv else 19.9
tot = {}
for f in sorted(glob.glob(os.path.join(d, "*-t*.full.log"))):
    name = os.path.basename(f)[:-9]
    s = [(float(m[1]), float(m[5])) for m in map(pat.search, open(f, errors="replace")) if m and float(m[1]) <= end]
    if not s:
        continue
    resets = [float(l.split("t=")[1].split("s")[0]) for l in open(f, errors="replace") if "RESET at" in l]
    resets = [r for r in resets if r < end]
    state = "up" if s[0][1] < -0.9 else "down"
    downs = 0 if state == "up" else 1; rises = 0; times = []; down_t = 0.0; up_s = 0.0; prev = s[0][0]
    for t, gz in s:
        dt = t - prev; prev = t
        if state == "up":
            up_s += dt
        tilt = math.degrees(math.acos(max(-1, min(1, -gz))))
        if state == "up" and tilt > 60:
            state = "down"; downs += 1; down_t = t
        elif state == "down" and gz < -0.9:
            state = "up"; rises += 1; times.append(round(t - down_t, 1))
    frac = up_s / max(s[-1][0], 1e-6)
    print(f"{name:40s} downs={downs} rises={rises} rise_s={times} upright={frac:.2f} end={state} resets={resets}")
    g = name.rsplit("-t", 1)[0].split("-L")[0]
    a = tot.setdefault(g, [0, 0, 0, [], []]); a[0] += downs; a[1] += rises; a[2] += 1; a[3] += times; a[4].append(frac)
print("--- totals per policy+spawn (all rows/types)")
for g, (dn, ri, n, ts, fr) in sorted(tot.items()):
    med = sorted(ts)[len(ts) // 2] if ts else None
    print(f"{g:24s} takes={n} down_states={dn} rises={ri} median_rise_s={med} mean_upright={sum(fr)/len(fr):.2f}")
