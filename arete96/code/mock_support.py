#!/usr/bin/env python3
"""mock_support.py: helpers for the item-96 MOCK proof only (never a result). Frozen.
  mock_support.py subset ITEMS OUT N        -> first N prompts of the eval (the mock run's items file)
  mock_support.py board RUNDIR REFDIR       -> RUNDIR/board/new-q6k/<task>.jsonl = the 90a reference rows with a few items flipped (stands in for the Arete board leg)
"""
import json, os, sys

cmd = sys.argv[1]
if cmd == "subset":
    rows = [l for l in open(sys.argv[2])][:int(sys.argv[4])]
    open(sys.argv[3], "w").writelines(rows)
    print(len(rows))
elif cmd == "board":
    run, ref = sys.argv[2], sys.argv[3]
    d = os.path.join(run, "board", "new-q6k")
    os.makedirs(d, exist_ok=True)
    for t, flips in (("gpqa", 3), ("he", 1), ("arc", 4), ("mmlu", 6)):
        rows = [json.loads(l) for l in open(os.path.join(ref, t + ".jsonl"))]
        with open(os.path.join(d, t + ".jsonl"), "w") as f:
            for i, r in enumerate(rows):
                if i < flips:
                    r = dict(r, ok=not r["ok"])
                f.write(json.dumps(r) + "\n")
    print("mock board written")
