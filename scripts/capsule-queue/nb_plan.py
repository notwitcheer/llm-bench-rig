#!/usr/bin/env python3
"""Planner for item 77: turns the measured fit ceilings into the KV type and context a leg runs with.

  nb_plan.py --fitdir D --need N --min M     -> "OK <kv> <ctx> <limited 0|1>" or "NOFIT <reason>"
  nb_plan.py --fitdir D --merge              -> writes D/fit.json {ceiling_f16, ceiling_q8_0, complete}; prints it

Rule (pre-registered in the item header): f16 KV whenever the leg's planned context N is at or under the f16 ceiling (the Qwen rows' regime);
else q8_0 KV at N if N is at or under the q8_0 ceiling (labelled deviation); else q8_0 KV at the q8_0 ceiling, flagged limited, provided that is at
least the leg's minimum viable context M; else NOFIT. A ceiling that is None (the first ladder point failed) means that KV type cannot run at all.
"""
import argparse
import json
import os


def load(d, kv):
    p = os.path.join(d, f"fit-{kv}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitdir", required=True)
    ap.add_argument("--need", type=int, default=0)
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    f16, q8 = load(a.fitdir, "f16"), load(a.fitdir, "q8_0")
    if a.merge:
        out = {"ceiling_f16": (f16 or {}).get("ceiling_ctx"), "ceiling_q8_0": (q8 or {}).get("ceiling_ctx"),
               "first_fail_f16": (f16 or {}).get("first_fail_ctx"), "first_fail_q8_0": (q8 or {}).get("first_fail_ctx"),
               "complete": bool(f16 and q8 and f16.get("complete") and q8.get("complete"))}
        json.dump(out, open(os.path.join(a.fitdir, "fit.json"), "w"), indent=1)
        print(json.dumps(out))
        return
    if not (f16 and q8):
        print("NOFIT fit leg not done")
        return
    c16, c8 = f16.get("ceiling_ctx") or 0, q8.get("ceiling_ctx") or 0
    if a.need <= c16:
        print(f"OK f16 {a.need} 0")
    elif a.need <= c8:
        print(f"OK q8_0 {a.need} 0")
    elif c8 >= a.min and c8 > 0:
        print(f"OK q8_0 {c8} 1")
    else:
        print(f"NOFIT ceilings f16 {c16} / q8_0 {c8} below the minimum viable context {a.min} (planned {a.need})")


if __name__ == "__main__":
    main()
