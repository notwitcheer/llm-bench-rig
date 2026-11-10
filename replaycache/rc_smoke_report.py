#!/usr/bin/env python3
"""rc_smoke_report.py <run_dir>: sizing numbers from a -smoke run (never counted): per-solve seconds / tokens / decode rate, the build loop, and the card-minute projection."""
import json, os, statistics, sys
run = sys.argv[1]
nfull_solves, nbuild = int(sys.argv[2]) if len(sys.argv) > 2 else 400, int(sys.argv[3]) if len(sys.argv) > 3 else 20
A = [json.load(open(os.path.join(run, "results/A", f))) for f in sorted(os.listdir(os.path.join(run, "results/A")))] if os.path.isdir(os.path.join(run, "results/A")) else []
S = [json.load(open(os.path.join(run, "results/build-src", f))) for f in sorted(os.listdir(os.path.join(run, "results/build-src")))] if os.path.isdir(os.path.join(run, "results/build-src")) else []
E = [json.load(open(os.path.join(run, "entries", f))) for f in sorted(os.listdir(os.path.join(run, "entries"))) if f.endswith(".json") and not f.endswith(".failed.json")] if os.path.isdir(os.path.join(run, "entries")) else []
out = {}
if A:
    out["source"] = {"n": len(A), "correct": sum(1 for x in A if x["correct"]), "tokens_mean": statistics.mean(x["completion_tokens"] for x in A), "tokens_max": max(x["completion_tokens"] for x in A),
                     "wall_s_mean": statistics.mean(x["wall_s"] for x in A), "wall_s_max": max(x["wall_s"] for x in A), "decode_tok_s_mean": statistics.mean(x["decode_tok_s"] or 0 for x in A),
                     "finish": [x["finish"] for x in A], "how": [x["how"] for x in A]}
if S:
    out["build_src"] = {"n": len(S), "tokens": [x["completion_tokens"] for x in S], "wall_s": [x["wall_s"] for x in S], "correct": [x["correct"] for x in S]}
if E:
    out["entries"] = [{"tid": e["tid"], "attempts": len(e["attempts"]), "builder_tokens": e["builder_tokens"], "builder_wall_s": e["builder_wall_s"], "n_params": len(e["schema"]), "source_matches_golden": e["source_matches_golden"]} for e in E]
if A:
    # production runs 4 slots at a time: wall per solve already includes the 4-way batching of this smoke when 4 ran together
    per = out["source"]["wall_s_mean"]
    proj_src = nfull_solves * per / 4 / 60
    bt = statistics.mean([s["wall_s"] for s in S]) if S else per
    bb = statistics.mean([e["builder_wall_s"] for e in E]) if E else per * 1.5
    proj_build = nbuild * (bt + bb) / 4 / 60
    out["projection_min"] = {"source": round(proj_src, 1), "build": round(proj_build, 1), "loads_drain_restore": 6, "total": round(proj_src + proj_build + 6, 1), "basis": "mean wall per solve %.0f s at 4 slots, %d solves, %d builds" % (per, nfull_solves, nbuild)}
json.dump(out, open(os.path.join(run, "smoke-report.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
