#!/usr/bin/env python3
"""hermes-settings paired reader (offline, CPU only): compare two toolcall-v1 leg dirs item by item.

usage: python3 hermes_settings_paired.py <base_leg_dir> <other_leg_dir> [--json out.json]
Each dir is a ~/benchmark-rig/results/.../<slug>-<leg> with items.jsonl (single-turn) and/or multi_turn.jsonl (S5).
Per slice: n paired ids, correct in each, discordant pairs (base-only / other-only), exact two-sided McNemar p,
and how many items came out byte-identical (single-turn: same generated text; S5: same step completion-token
sequence and same decoded calls). Identical-output rate is the run-to-run determinism check: two runs of the same
config should sit at ~100 %, and a KV-type or ctx change is read against that.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def load_jsonl(p):
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    return out


def single(d):
    rows = {}
    for r in load_jsonl(d / "items.jsonl"):
        rows[r["id"]] = r  # last wins (resume duplicates)
    return rows


def multi(d):
    rows = {}
    for r in load_jsonl(d / "multi_turn.jsonl"):
        rows[r["id"]] = r
    return rows


def compare(name, a, b, ok, same):
    ids = sorted(set(a) & set(b))
    ca = {i: ok(a[i]) for i in ids}
    cb = {i: ok(b[i]) for i in ids}
    bo = sum(1 for i in ids if ca[i] and not cb[i])
    oo = sum(1 for i in ids if cb[i] and not ca[i])
    ident = sum(1 for i in ids if same(a[i], b[i]))
    return {"slice": name, "paired": len(ids), "only_in_base": len(set(a) - set(b)), "only_in_other": len(set(b) - set(a)),
            "base_correct": sum(ca.values()), "other_correct": sum(cb.values()), "base_only_correct": bo, "other_only_correct": oo,
            "mcnemar_exact_p": round(mcnemar_exact(bo, oo), 4), "identical_outputs": ident,
            "identical_pct": round(100 * ident / len(ids), 1) if ids else None,
            "base_completion_tokens_mean": round(sum(_ct(a[i]) for i in ids) / len(ids), 1) if ids else None,
            "other_completion_tokens_mean": round(sum(_ct(b[i]) for i in ids) / len(ids), 1) if ids else None}


def _ct(r):
    return r.get("completion_tokens") or 0


def steps_sig(r):
    return ([s.get("completion_tokens") for s in r.get("steps_log", [])], json.dumps(r.get("decoded"), sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--json")
    a = ap.parse_args()
    base, other = Path(a.base), Path(a.other)
    res = []
    sa, sb = single(base), single(other)
    if sa and sb:
        res.append(compare("single-turn S1..S4 (unique requests)", sa, sb, lambda r: r.get("outcome") == "correct",
                           lambda x, y: x.get("generated") == y.get("generated")))
    ma, mb = multi(base), multi(other)
    if ma and mb:
        res.append(compare("S5 multi_turn_base", ma, mb, lambda r: r.get("outcome") == "correct",
                           lambda x, y: steps_sig(x) == steps_sig(y)))
    if not res:
        sys.exit("nothing to compare: need items.jsonl or multi_turn.jsonl in both dirs")
    for r in res:
        print(json.dumps(r))
    if a.json:
        Path(a.json).write_text(json.dumps({"base": str(base), "other": str(other), "slices": res}, indent=1))


if __name__ == "__main__":
    main()
