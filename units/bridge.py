#!/usr/bin/env python3
"""bridge leg (OPTIONAL): does OUR runtime reproduce the paper's per-item answers on the same files? 100 `unit` views + 100 `fraction` views from the paper's frozen
views_benchmark.jsonl, the paper's exact request (one user message, chat_template_kwargs enable_thinking=false, temperature 0, top_k 1, top_p 1, min_p 0, seed 20260725,
max_tokens 1024, no cache_prompt override), answers parsed with the paper's own strict parser and compared item by item with the paper's eval.jsonl for the same system.
selection (fixed in PREREG.md before any output): shuffle all views with random.Random(20260725) as the paper's runner does, take the first 100 unit and the first 100 fraction views in that order.
the paper's artefacts are NOT part of the public clone: --anc points at a copy of the arXiv 2609.25009v1 ancillary archive (views + eval.jsonl + src/num_equiv).
usage: bridge.py --anc DIR --system qwen35-9b-q4 --port N --out DIR [--n 100] [--deadline-epoch E]     |     bridge.py --compare --anc DIR --system S --out DIR
exit: 0 done | 3 deadline | 4 server unhealthy"""
import argparse
from fractions import Fraction
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260725


def read_jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def select(anc, n):
    views = read_jsonl(os.path.join(anc, "data/generated/views_benchmark.jsonl"))
    order = list(range(len(views)))
    random.Random(SEED).shuffle(order)
    got = {"unit": [], "fraction": []}
    for i in order:
        v = views[i]
        if v["view_id"] in got and len(got[v["view_id"]]) < n:
            got[v["view_id"]].append(v)
    return [views[i] for i in order if views[i] in got.get(views[i]["view_id"], [])]


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def paper_parse(anc):
    sys.path.insert(0, os.path.join(anc, "src"))
    from num_equiv.parse import parse_response          # the paper's strict parser, unchanged
    return parse_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anc", required=True)
    ap.add_argument("--system", required=True, help="qwen35-9b-q4 | qwen35-9b-q8 (directory name under artifacts/full/)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--deadline-epoch", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bridge-%s.jsonl" % a.system)
    parse_response = paper_parse(a.anc)
    if a.compare:
        return compare(a, path, parse_response)
    items = select(a.anc, a.n)
    done = {(r["base_id"], r["view_id"]) for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [v for v in items if (v["base_id"], v["view_id"]) not in done]
    base = "http://127.0.0.1:%d" % a.port
    try:
        h = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read().decode())
        assert "ok" in json.dumps(h).lower()
    except Exception as e:
        print("[bridge] server not healthy", repr(e))
        return 4
    print("[bridge] %s: %d selected, %d done, %d to do" % (a.system, len(items), len(done), len(todo)), flush=True)
    n_new, t0 = 0, time.time()
    with open(path, "a") as out:
        for v in todo:
            if a.deadline_epoch and time.time() >= a.deadline_epoch:
                print("[bridge] deadline reached after %d rows" % n_new, flush=True)
                return 3
            payload = {"model": a.system, "messages": [{"role": "user", "content": v["prompt"]}], "chat_template_kwargs": {"enable_thinking": False},
                       "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": SEED, "max_tokens": 1024, "stream": False}
            t1 = time.time()
            body = post(base + "/v1/chat/completions", payload)
            ch = body["choices"][0]
            content, fin = ch["message"].get("content") or "", ch.get("finish_reason", "stop")
            o = parse_response(content, v["dimension"], fin)
            cat = ("correct" if o.value == Fraction(v["expected_number"]) else "value_error") if o.category == "ok" else o.category   # same mapping as the paper's evaluate.py
            out.write(json.dumps({"base_id": v["base_id"], "view_id": v["view_id"], "category": cat, "parsed_value": str(o.value) if o.value is not None else None,
                                  "finish_reason": fin, "completion_tokens": (body.get("usage") or {}).get("completion_tokens"), "latency_s": round(time.time() - t1, 3)}) + "\n")
            out.flush()
            n_new += 1
    print("[bridge] %s complete: %d new rows in %.0f s" % (a.system, n_new, time.time() - t0), flush=True)
    return 0


def compare(a, path, parse_response):
    ours = {(r["base_id"], r["view_id"]): r for r in read_jsonl(path)}
    paper = {(r["base_id"], r["view_id"]): r for r in read_jsonl(os.path.join(a.anc, "artifacts/full/%s/eval.jsonl" % a.system))}
    rows = {"unit": [0, 0, 0, 0], "fraction": [0, 0, 0, 0]}    # n, same parsed value, same category, both correct
    diffs = []
    for k, r in ours.items():
        p = paper.get(k)
        if p is None:
            continue
        x = rows[k[1]]
        x[0] += 1
        sv = (r["parsed_value"] == p["parsed_value"]) or (r["parsed_value"] is not None and p["parsed_value"] is not None and Fraction(r["parsed_value"]) == Fraction(p["parsed_value"]))
        x[1] += 1 if sv else 0
        x[2] += 1 if r["category"] == p["category"] else 0
        x[3] += 1 if (r["category"] == "correct" and p["correct"]) else 0
        if not sv:
            diffs.append({"key": list(k), "ours": [r["category"], r["parsed_value"]], "paper": [p["category"], p["parsed_value"]]})
    out = {"system": a.system, "per_view": {v: {"n": x[0], "same_parsed_value": x[1], "same_category": x[2], "both_correct": x[3]} for v, x in rows.items()}, "n_diff": len(diffs), "diffs": diffs[:40],
           "note": "runtime-parity check against the paper's eval.jsonl (llama.cpp b10107 on an H100); not a result about tool arguments"}
    json.dump(out, open(os.path.join(a.out, "bridge-%s-compare.json" % a.system), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "diffs"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
