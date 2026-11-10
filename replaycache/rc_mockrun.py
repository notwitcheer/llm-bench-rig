#!/usr/bin/env python3
"""rc_mockrun.py <out_run_dir> <sel.json>: writes a synthetic (MOCK) run directory for the selection's templates, with no servers: arm A / B / C records, entries and card.json.
Used by the item's DRYRUN to prove summary + render on CPU. Never counted: every record carries mock=true."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as C

run, selp = sys.argv[1], sys.argv[2]
sel = json.load(open(selp))
rng = random.Random(7)
MAIN = C.index_rows("main")
for t in sel["templates"]:
    tid = t["tid"]
    q0 = MAIN[(tid, 0)]
    C.atomic_json(os.path.join(run, "entries", "%d.json" % tid), {"tid": tid, "description": "mock family %d" % tid, "schema": [{"name": "x", "description": "a number", "example": 1}], "build_params": {"x": 1},
                  "build_question": q0["question"], "source_answer": C.golden_of(q0), "golden_logged_only": C.golden_of(q0), "source_matches_golden": True, "attempts": [{"attempt": 0, "ok": True}],
                  "builder_tokens": 3000, "builder_wall_s": 90.0, "code_file": "%d.py" % tid, "mock": True})
    open(os.path.join(run, "entries", "%d.py" % tid), "w").write("def solve(params):\n    return params['x']\n")
emb_rows = []
for kind in ("held", "near"):
    for (_k, tid, inst) in C.items_of(sel, kind):
        key = C.item_key(kind, tid, inst)
        row = C.row_for(kind, tid, inst); g = C.golden_of(row)
        ntok = int(rng.uniform(300, 4000)); wall = ntok / 11.0
        okA = rng.random() < 0.92
        C.atomic_json(os.path.join(run, "results", "A", key + ".json"), {"key": key, "kind": kind, "tid": tid, "inst": inst, "golden": g, "answer": g if okA else g + 1, "correct": okA, "how": "marker",
                      "content": "#### %g" % g, "reasoning": "mock reasoning about %s" % row["question"][:50], "finish": "stop", "completion_tokens": ntok, "prompt_tokens": 80, "wall_s": round(wall, 1), "decode_tok_s": 45.0, "mock": True})
        s = rng.uniform(0.52, 0.72) if kind == "held" else rng.uniform(0.50, 0.70)
        served = rng.random() < 0.97
        okB = (rng.random() < 0.95) if kind == "held" else (rng.random() < 0.15)
        val = g if okB else g + 3
        C.atomic_json(os.path.join(run, "cpu", "B", key + ".json"), {"key": key, "kind": kind, "tid": tid, "inst": inst, "entry": tid, "s": s, "params": {"x": 1}, "extract_ok": served, "value": (val if served else None),
                      "served": served, "why": "" if served else "mock miss", "golden": g, "correct": bool(served and okB), "extract_ms": 600.0, "solve_ms": 12.0, "embed_ms": 40.0, "cpu_ms": 650.0,
                      "lm_tokens": 30, "lm_prompt_tokens": 400, "mock": True})
        okC = rng.random() < 0.25
        C.atomic_json(os.path.join(run, "cpu", "C", key + ".json"), {"key": key, "kind": kind, "tid": tid, "inst": inst, "golden": g,
                      "C": {"answer": g if okC else 1.0, "how": "marker", "correct": okC, "tokens": 180, "wall_ms": 4200.0, "finish": "stop"},
                      "Cd": {"answer": 1.0, "how": "last_number", "correct": False, "tokens": 3, "wall_ms": 300.0, "finish": "stop"}, "mock": True})
C.atomic_json(os.path.join(run, "card.json"), {"card_min": 0.0, "mock": True})
open(os.path.join(run, "sel.json"), "w").write(json.dumps(sel))
print("mock run written:", run)
