#!/usr/bin/env python3
"""prove every v2 checker BEFORE any model output exists.

per task: (1) fresh seed must FAIL  (2) near miss must FAIL  (3) hand solution must PASS with frac 1.0
        (4) the seed is deterministic and path independent (same tree hash twice and under a second work root)
        (5) an empty task dir must FAIL (checker must not crash or pass on nothing)
writes results/checker-proof-<stamp>.json (or --out). exit 0 only if every task proves.

usage: prove_checkers.py [--work /opt/data/tmp/at2-proof] [--out FILE] [--only t01,t02]"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def guard(work):
    work = os.path.abspath(work)
    base = os.path.basename(work)
    if not (base.startswith("at2-proof") or base.startswith("proof")) or work.count("/") < 3:
        sys.exit(f"refusing to wipe {work}: the proof work dir name must start with at2-proof")
    return work


def run(work, only=None):
    work = guard(work)
    rows = []
    for cls in common.loaded_tasks():
        if only and cls.id not in only:
            continue
        T = os.path.join(work, cls.id)
        res = {"id": cls.id, "name": cls.name, "cat": cls.cat, "diff": cls.diff}
        try:
            common.rm_tree(T)
            os.makedirs(T)
            res["empty"] = list(cls.check(T))
            common.rm_tree(T)
            cls.seed(T)
            h1 = common.tree_hash(T)
            res["fresh"] = list(cls.check(T))
            common.rm_tree(T)
            cls.seed(T)
            h2 = common.tree_hash(T)
            T2 = os.path.join(work + "-b", cls.id)
            common.rm_tree(T2)
            cls.seed(T2)
            h3 = common.tree_hash(T2)
            common.rm_tree(os.path.join(work + "-b"))
            res["seed_hash"] = h1
            res["seed_deterministic"] = h1 == h2 == h3
            common.rm_tree(T)
            cls.seed(T)
            cls.near(T)
            res["near_miss"] = list(cls.check(T))
            common.rm_tree(T)
            cls.seed(T)
            cls.solve(T)
            res["hand_solution"] = list(cls.check(T))
            ok = (not res["empty"][0]) and (not res["fresh"][0]) and (not res["near_miss"][0]) and res["hand_solution"][0] \
                and res["hand_solution"][2] == 1.0 and res["seed_deterministic"]
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"
            ok = False
        res["proof_ok"] = bool(ok)
        rows.append(res)
        print(f"{cls.id} {cls.diff} {cls.cat:9s} empty={'FAIL' if res.get('empty') and not res['empty'][0] else '??'} "
              f"fresh={'FAIL' if res.get('fresh') and not res['fresh'][0] else 'PASS!'} "
              f"near={'FAIL' if res.get('near_miss') and not res['near_miss'][0] else 'PASS!'} "
              f"hand={'PASS' if res.get('hand_solution') and res['hand_solution'][0] else 'FAIL!'} "
              f"det={res.get('seed_deterministic')} {'OK' if ok else 'BROKEN ' + res.get('error', '')}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/opt/data/tmp/at2-proof")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    rows = run(a.work, set(a.only.split(",")) if a.only else None)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_ok = sum(r["proof_ok"] for r in rows)
    res = {"stamp_utc": stamp, "work": a.work, "n": len(rows), "n_proved": n_ok, "all_ok": n_ok == len(rows), "tasks": rows}
    out = a.out or os.path.join(HERE, "results", f"checker-proof-{stamp[:10]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(("ALL_OK " if res["all_ok"] else "PROOF_BROKEN ") + f"{n_ok}/{len(rows)}", out)
    sys.exit(0 if res["all_ok"] else 1)
