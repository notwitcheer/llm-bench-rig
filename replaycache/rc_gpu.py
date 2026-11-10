#!/usr/bin/env python3
"""rc_gpu.py: the card legs of the replay-cache item. Talks to ONE llama-server (house model, own port) and writes resumable per-item files.
  source  leg: arm A, one from-scratch completion per question (held-out main + near-miss p1)      -> <out>/results/A/<key>.json
  build   leg: per template, the source solves the build instance (instance 0 of main), then the builder writes solve()+schema; validated against the
               SOURCE's answer (never the golden label), max 3 rewrites                                -> <out>/results/build-src/<tid>.json, <out>/entries/<tid>.{py,json}
rc: 0 all pending items done | 3 items remain (deadline/window) | 1 errors.
"""
import argparse, concurrent.futures as cf, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_common import *

ap = argparse.ArgumentParser()
ap.add_argument("leg", choices=["source", "build"])
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sel", required=True)
ap.add_argument("--deadline", type=float, required=True)
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=20000)
ap.add_argument("--limit-templates", type=int, default=0)
ap.add_argument("--limit-items", type=int, default=0, help="smoke: only the first N source items")
ap.add_argument("--model", default="qwen38-27b")
ap.add_argument("--count-only", action="store_true", help="print the pending count of this leg and exit 3 when anything is pending, 0 otherwise")
ap.add_argument("--min-tail-min", type=float, default=14.0, help="no new request starts when fewer minutes than this remain")
a = ap.parse_args()

sel = json.load(open(a.sel))
LT = a.limit_templates or None
os.makedirs(a.out, exist_ok=True)
lock = threading.Lock()
stop_launch = a.deadline - a.min_tail_min * 60
remaining = []


def item_list():
    return [(idx, kind, tid, inst) for idx, (kind, tid, inst) in enumerate(limited_items(sel, LT, a.limit_items))]


def do_source(it):
    idx, kind, tid, inst = it
    key = item_key(kind, tid, inst)
    path = os.path.join(a.out, "results", "A", key + ".json")
    if os.path.exists(path):
        return "have"
    if time.time() > stop_launch:
        with lock:
            remaining.append(key)
        return "skipped"
    row = row_for(kind, tid, inst)
    t0 = time.time()
    try:
        rec = solve_one(a.base, row["question"], 1000 + idx, a.max_tokens, a.model, timeout=max(60, a.deadline - time.time() - 60))
    except Exception as e:
        log("source error", key, repr(e)[:120])
        with lock:
            remaining.append(key)
        return "error"
    g = golden_of(row)
    rec.update({"key": key, "kind": kind, "tid": tid, "inst": inst, "golden": g, "correct": same(rec["answer"], g), "t_end": time.time()})
    atomic_json(path, rec)
    log("A %s ans=%s gold=%s %s tok=%s %.0fs" % (key, rec["answer"], g, "ok" if rec["correct"] else "WRONG", rec["completion_tokens"], rec["wall_s"]))
    return "done"


def build_one(t):
    try:
        return build_one_inner(t)
    except Exception as e:
        log("build error", t["tid"], repr(e)[:160])
        with lock:
            remaining.append(str(t["tid"]))
        return "error"


def build_one_inner(t):
    tid = t["tid"]
    ent_json = os.path.join(a.out, "entries", "%d.json" % tid)
    fail_json = os.path.join(a.out, "entries", "%d.failed.json" % tid)
    if os.path.exists(ent_json) or os.path.exists(fail_json):
        return "have"
    if time.time() > stop_launch:
        with lock:
            remaining.append(str(tid))
        return "skipped"
    row = index_rows_cached()[(tid, 0)]
    q = row["question"]; gold = golden_of(row)
    sp = os.path.join(a.out, "results", "build-src", "%d.json" % tid)
    if os.path.exists(sp):
        src = json.load(open(sp))
    else:
        src = solve_one(a.base, q, 0, a.max_tokens, a.model, timeout=max(60, a.deadline - time.time() - 60))
        src.update({"tid": tid, "golden": gold, "correct": same(src["answer"], gold)})
        atomic_json(sp, src)
    log("build-src %d ans=%s gold=%s tok=%s %.0fs" % (tid, src["answer"], gold, src["completion_tokens"], src["wall_s"]))
    if src["answer"] is None:
        atomic_json(fail_json, {"tid": tid, "why": "the source produced no parsable answer", "golden": gold})
        return "failed"
    want = src["answer"]
    msgs = [{"role": "user", "content": BUILD_PROMPT.format(question=q, answer=("%g" % want), transcript=transcript_of(src))}]
    attempts, tok, wall = [], 0, 0.0
    for att in range(4):                       # initial + 3 rewrites
        if time.time() > a.deadline - 120:
            with lock:
                remaining.append(str(tid))
            return "deadline"
        r = chat(a.base, msgs, a.max_tokens, model=a.model, seed=att, timeout=max(60, a.deadline - time.time() - 90), chat_template_kwargs={"enable_thinking": True})
        content, reasoning = msg_of(r)
        tok += r.get("usage", {}).get("completion_tokens", 0); wall += r["_wall_s"]
        parsed, why = parse_builder(content)
        got = None
        if parsed:
            ok, got, why = validate_entry(parsed, os.path.join(a.out, "work", str(tid)), want, "a%d" % att)
        else:
            ok = False
        attempts.append({"attempt": att, "ok": ok, "why": why, "got": got, "completion_tokens": r.get("usage", {}).get("completion_tokens")})
        log("build %d attempt %d ok=%s %s" % (tid, att, ok, why))
        if ok:
            ed = os.path.join(a.out, "entries"); os.makedirs(ed, exist_ok=True)
            open(os.path.join(ed, "%d.py" % tid), "w").write(parsed["code"])
            atomic_json(ent_json, {"tid": tid, "description": parsed["description"], "schema": parsed["schema"], "build_params": parsed["params"], "build_question": q,
                                   "source_answer": want, "golden_logged_only": gold, "source_matches_golden": same(want, gold), "attempts": attempts, "builder_tokens": tok,
                                   "builder_wall_s": round(wall, 1), "code_file": "%d.py" % tid})
            return "built"
        msgs = msgs + [{"role": "assistant", "content": content}, {"role": "user", "content": BUILD_FIX.format(got=("%g" % got) if got is not None else "nothing", answer=("%g" % want), why=why)}]
    atomic_json(fail_json, {"tid": tid, "why": "no validated entry after 4 attempts", "attempts": attempts, "golden_logged_only": gold, "source_answer": want})
    return "failed"


_idx = {}


def index_rows_cached():
    if "m" not in _idx:
        _idx["m"] = index_rows("main")
    return _idx["m"]


if a.count_only:
    if a.leg == "source":
        pend = [it for it in item_list() if not os.path.exists(os.path.join(a.out, "results", "A", item_key(it[1], it[2], it[3]) + ".json"))]
        tot = len(item_list())
    else:
        ts = sel["templates"][:LT] if LT else sel["templates"]
        pend = [t for t in ts if not (os.path.exists(os.path.join(a.out, "entries", "%d.json" % t["tid"])) or os.path.exists(os.path.join(a.out, "entries", "%d.failed.json" % t["tid"])))]
        tot = len(ts)
    print("PENDING %s %d of %d" % (a.leg, len(pend), tot))
    sys.exit(3 if pend else 0)
t0 = time.time()
if a.leg == "source":
    items = item_list()
    with cf.ThreadPoolExecutor(a.slots) as ex:
        res = list(ex.map(do_source, items))
else:
    ts = sel["templates"][:LT] if LT else sel["templates"]
    with cf.ThreadPoolExecutor(a.slots) as ex:
        res = list(ex.map(build_one, ts))
from collections import Counter
c = Counter(res)
log("[rc_gpu] leg %s end: %s, %d remaining, %.1f min" % (a.leg, dict(c), len(remaining), (time.time() - t0) / 60))
sys.exit(3 if remaining else (1 if c.get("error") else 0))
