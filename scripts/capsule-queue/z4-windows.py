#!/usr/bin/env python3
# 100-it window means of mean reward, fell_over, error_vel_xy from an rsl_rl run log.
import re, sys
f, ends = sys.argv[1], [int(x) for x in sys.argv[2].split(',')]
t = open(f, errors='replace').read()
blocks = re.split(r'Learning iteration (\d+)/\d+', t)
rows = {}
for i in range(1, len(blocks), 2):
    it = int(blocks[i]); b = blocks[i + 1]
    def g(k):
        m = re.search(re.escape(k) + r':\s*([-\d.eE+]+)', b)
        return float(m[1]) if m else None
    rows[it] = (g('Mean reward'), g('Episode_Termination/fell_over'), g('Metrics/twist/error_vel_xy'))
for e in ends:
    w = [rows[k] for k in range(e - 100, e) if k in rows]
    out = []
    for j in range(3):
        v = [x[j] for x in w if x[j] is not None]
        out.append(sum(v) / len(v) if v else float('nan'))
    print(f"..{e}: n={len(w)} reward {out[0]:.1f} fell_over {out[1]:.3f} err_xy {out[2]:.3f}")
