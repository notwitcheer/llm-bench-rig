#!/usr/bin/env python3
"""prove every checker FAILS on fresh seeds, FAILS on a plausible near miss, and PASSES on the hand
solution. run BEFORE any model output. writes results/checker-proof-<stamp>.json.

usage: prove_checkers.py [--work /opt/data/tmp/sparkagent/work] [--out results/checker-proof.json]"""
import argparse
import datetime
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "checkers"))
import gen_seeds  # noqa: E402
import hand_solutions  # noqa: E402


def run(work):
    rows = []
    for tid in gen_seeds.TASK_IDS:
        chk = importlib.import_module(f"check_{tid}").check
        gen_seeds.build(work)
        fresh = chk(work)
        hand_solutions.near_miss(tid, work)
        miss = chk(work)
        gen_seeds.build(work)
        hand_solutions.solve(tid, work)
        hand = chk(work)
        ok = (not fresh[0]) and (not miss[0]) and hand[0]
        rows.append({"id": tid, "fresh_seed": list(fresh), "near_miss": list(miss), "hand_solution": list(hand), "proof_ok": ok})
        print(f"{tid} fresh={'FAIL' if not fresh[0] else 'PASS!'} near_miss={'FAIL' if not miss[0] else 'PASS!'} "
              f"hand={'PASS' if hand[0] else 'FAIL!'}  {'OK' if ok else 'BROKEN'}")
    gen_seeds.build(work)  # leave a fresh tree behind
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=gen_seeds.default_work())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = run(a.work)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = {"stamp_utc": stamp, "work": a.work, "all_ok": all(r["proof_ok"] for r in rows), "tasks": rows}
    out = a.out or os.path.join(HERE, "results", f"checker-proof-{stamp[:10]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("ALL_OK" if res["all_ok"] else "PROOF_BROKEN", out)
    sys.exit(0 if res["all_ok"] else 1)
