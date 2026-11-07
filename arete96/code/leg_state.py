#!/usr/bin/env python3
"""leg_state.py: small CPU helpers for the item-96 PLAN phase and leg bookkeeping. Frozen (FROZEN.sha256).

  leg_state.py items-check ITEMS                      -> "ITEMS OK 803" (803 rows, unique ids, one id width) or "ITEMS BAD <why>"
  leg_state.py gen-pending RUN TAG ITEMS [LIMIT]      -> int: prompts (first LIMIT) with no row in RUN/completions-TAG.jsonl
  leg_state.py board-pending RUN [LIMIT]              -> int: board items still to run (the 90a reference rows set the expected count per task; first LIMIT per task when LIMIT given)
  leg_state.py board-identity                         -> "IDENTITY OK" when the board items (same loader as 90a, offline datasets) have exactly the keys and expected answers of the 90a reference rows
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
TASKS = ["gpqa", "he", "arc", "mmlu"]


def rl(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "items-check":
        items = [json.loads(l) for l in open(sys.argv[2])]
        ids = [i["id"] for i in items]
        if len(items) != 803:
            print("ITEMS BAD rows=%d" % len(items))
        elif len(set(ids)) != 803:
            print("ITEMS BAD duplicate ids")
        elif len({len(i) for i in ids}) != 1:
            print("ITEMS BAD id widths differ")
        elif any((not i.get("required_facts")) or (not i.get("prompt")) for i in items):
            print("ITEMS BAD empty prompt or facts")
        else:
            print("ITEMS OK 803")
    elif cmd == "gen-pending":
        run, tag, items = sys.argv[2], sys.argv[3], sys.argv[4]
        lim = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        ids = [json.loads(l)["id"] for l in open(items)]
        if lim:
            ids = ids[:lim]
        have = {r["id"] for r in rl(os.path.join(run, "completions-%s.jsonl" % tag)) if "id" in r}
        print(sum(1 for i in ids if i not in have))
    elif cmd == "board-pending":
        run = sys.argv[2]
        lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        miss = 0
        for t in TASKS:
            want = len(rl(os.path.join(ROOT, "ref90a", t + ".jsonl")))
            if lim:
                want = min(want, lim)
            have = {r["k"] for r in rl(os.path.join(run, "board", "new-q6k", t + ".jsonl")) if "k" in r}
            miss += max(0, want - len(have))
        print(miss)
    elif cmd == "board-identity":
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import nerf_common as N
        bad = []
        for t in TASKS:
            ref = {r["k"]: r for r in rl(os.path.join(ROOT, "ref90a", t + ".jsonl"))}
            got = {it["key"]: it for it in N.load_items(t)}
            if set(ref) != set(got):
                bad.append("%s keys differ (ref %d, now %d)" % (t, len(ref), len(got)))
                continue
            if t != "he":
                n = sum(1 for k in ref if ref[k].get("e") != got[k]["expected"])
                if n:
                    bad.append("%s: %d expected answers differ" % (t, n))
        print("IDENTITY OK" if not bad else "IDENTITY BAD " + "; ".join(bad[:3]))
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
