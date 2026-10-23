#!/usr/bin/env python3
"""Quality screen for the Humming legs via the rig's own evaluators over /v1.
usage: hm_quality.py <port> <leg> <outdir> [limit]"""
import sys, json, time
from pathlib import Path
RIG = Path.home() / "benchmark-rig"
sys.path.insert(0, str(RIG))
from lib.quality import _run_evals  # noqa: E402

port, leg, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 200
outdir.mkdir(parents=True, exist_ok=True)
api = f"http://127.0.0.1:{port}/v1"
t0 = time.time()
res = _run_evals(api, sys.argv[5] if len(sys.argv) > 5 else "hm", ["gsm8k", "arc_challenge"], results_dir=outdir,
                 sample=None, think=False, limit=limit, mc_gate_tokens=50)
summary = {"leg": leg, "limit": limit, "think": False, "results": res, "wall_s": round(time.time() - t0, 1)}
print(leg, "QUALITY", json.dumps(summary), flush=True)
json.dump(summary, open(outdir / "quality_summary.json", "w"), indent=1)
