#!/usr/bin/env python3
"""freeze_offline.py: write offline/FROZEN.sha256 (sha256sum -c format, paths relative to the agenttest code root). The item checks it at every DRYRUN and every start.
covers: the v1 frozen task set (tasks.json, seed_data, checkers, gen_seeds, hand_solutions), the v2 pieces the driver imports (run_agent, common, parse_server_log) and offline/*.py.
re-run after ANY deliberate edit; an unexplained mismatch is the point of the check."""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rels = ["tasks.json", "gen_seeds.py", "hand_solutions.py", "v2/run_agent.py", "v2/common.py", "v2/parse_server_log.py"]
for sub, ext in (("checkers", ".py"), ("seed_data", None), ("offline", ".py")):
    for dp, dn, fns in os.walk(os.path.join(ROOT, sub)):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for fn in fns:
            if (ext is None or fn.endswith(ext)) and fn != "FROZEN.sha256":
                rels.append(os.path.relpath(os.path.join(dp, fn), ROOT))
lines = [f"{hashlib.sha256(open(os.path.join(ROOT, r), 'rb').read()).hexdigest()}  {r}" for r in sorted(set(rels))]
open(os.path.join(ROOT, "offline", "FROZEN.sha256"), "w").write("\n".join(lines) + "\n")
print(len(lines), "files frozen")
