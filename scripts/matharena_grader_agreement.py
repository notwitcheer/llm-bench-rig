"""Agreement of the rig's strict maths grader with MathArena's own labels.

MathArena publishes every graded answer of the models on its leaderboard (MathArena/<set>_outputs:
gold_answer, parsed_answer, correct). Re-grade their parsed answers with lib.evals.matharena.grade_with_timeout
and count how often our verdict matches their label. For ArXivMath their label comes from an LLM judge
(mathematical equivalence), so disagreement = where a strict sympy grader under- or over-counts.

usage (capsule, rig venv):  PYTHONPATH=. .venv/bin/python scripts/matharena_grader_agreement.py [set ...]
"""
import collections
import sys

from datasets import load_dataset

from lib.evals.matharena import grade_with_timeout

sets = sys.argv[1:] or ["arxivmath-0826", "arxivmath-0226", "aime_2026", "hmmt_feb_2026"]
for s in sets:
    name = f"MathArena/{s}_outputs"
    try:
        ds = load_dataset(name, split="train")
    except Exception as e:
        print(name, "LOAD FAIL", str(e)[:200], flush=True)
        continue
    conf = collections.Counter(); timeouts = 0; ex = []
    for r in ds:
        gold = str(r["gold_answer"]); lab = bool(r["correct"])
        pa = r.get("parsed_answer")
        pa = None if pa in (None, "", "None") else str(pa)
        ours, to = grade_with_timeout(pa, gold, secs=10)
        timeouts += to
        conf[(lab, ours)] += 1
        if ours != lab and len(ex) < 8:
            ex.append((gold[:70], (pa or "")[:70], lab))
    tot = sum(conf.values()); agree = conf[(True, True)] + conf[(False, False)]
    print(f"{s}: rows {tot}, agree {agree/tot:.1%}, judge-true/ours-false {conf[(True, False)]}, "
          f"judge-false/ours-true {conf[(False, True)]}, grade timeouts {timeouts}", flush=True)
    for g, p, l in ex:
        print(f"   gold {g!r} | parsed {p!r} | judge {l}", flush=True)
