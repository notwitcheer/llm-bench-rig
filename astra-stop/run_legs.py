#!/usr/bin/env python3
"""run_legs.py: one leg = one arm on a running llama-server (--parallel 4), a list of seeds (pelican) or problem ids (maths).
Resumable per try (results/<arm>/<kind>-<id>.json). A try is only started with >= START_MARGIN s before the deadline; tries still running
at deadline - STOP_MARGIN s are aborted and NOT saved (censored tries are never counted). rc 0 = every item has a result, 3 = items remain."""
import argparse, json, os, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astra_common import *

START_MARGIN = 15 * 60
STOP_MARGIN = 5 * 60


def run_try(base, kind, item, prompt, seed, max_tok, outdir, tag, stopflag, mathref):
    t0 = time.time(); think, ans = [], []; tt, ta = [], []; fin = None; usage = None; timings = None
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(chat_body(prompt, seed, max_tok)).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    for raw in resp:
        if stopflag.is_set():
            resp.close(); return None
        raw = raw.strip()
        if not raw.startswith(b"data:") or raw == b"data: [DONE]":
            continue
        ch = json.loads(raw[5:])
        if ch.get("usage"): usage = ch["usage"]
        if ch.get("timings"): timings = ch["timings"]
        for c in ch.get("choices", []):
            d = c.get("delta", {}); ms = int((time.time() - t0) * 1000)
            if d.get("reasoning_content"): think.append(d["reasoning_content"]); tt.append(ms)
            if d.get("content"): ans.append(d["content"]); ta.append(ms)
            if c.get("finish_reason"): fin = c["finish_reason"]
    wall = time.time() - t0
    think_s, ans_s = "".join(think), "".join(ans)
    r = dict(tag=tag, kind=kind, id=item, seed=seed, finish_reason=fin, wall_s=round(wall, 1),
             think_tokens=tokenize_len(base, think_s), think_chunks=len(tt), answer_tokens=tokenize_len(base, ans_s), answer_chunks=len(ta),
             completion_tokens=(usage or {}).get("completion_tokens"), prompt_tokens=(usage or {}).get("prompt_tokens"),
             think_end_ms=(tt[-1] if tt else None), answer_end_ms=(ta[-1] if ta else None),
             decode_tps_server=(timings or {}).get("predicted_per_second"))
    if kind == "pelican":
        svg = extract_svg(ans_s); chk = svg_check(svg)
        r["svg"] = chk; r["complete"] = bool(fin == "stop" and chk["closed"] and chk["parses"] and chk["renders"])
    else:
        b = boxed(ans_s); r["boxed"] = b; r["correct"] = (norm_ans(b) is not None and norm_ans(b) == norm_ans(mathref))
        r["complete"] = bool(fin == "stop" and b is not None)
    base_p = os.path.join(outdir, f"{kind}-{item}")
    open(base_p + ".think.txt", "w").write(think_s); open(base_p + ".answer.txt", "w").write(ans_s)
    json.dump(dict(think_ms=tt, answer_ms=ta), open(base_p + ".tok.json", "w"))
    json.dump(r, open(base_p + ".json", "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--arm", required=True); ap.add_argument("--kind", choices=["pelican", "maths"], required=True)
    ap.add_argument("--items", required=True, help="comma list of seeds (pelican) or MATH-500 row ids (maths)")
    ap.add_argument("--out", required=True); ap.add_argument("--deadline", type=float, required=True, help="epoch")
    ap.add_argument("--slots", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--math-file", default="")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"; outdir = os.path.join(a.out, a.arm); os.makedirs(outdir, exist_ok=True)
    items = [int(x) for x in a.items.split(",")]
    rows = []
    if a.kind == "maths":
        rows = [json.loads(l) for l in open(a.math_file)]
    todo = [i for i in items if not os.path.exists(os.path.join(outdir, f"{a.kind}-{i:02d}.json" if a.kind == "pelican" else f"{a.kind}-{i:03d}.json"))]
    print(f"[run_legs] arm={a.arm} kind={a.kind} items={len(items)} todo={len(todo)} deadline in {int(a.deadline - time.time()) // 60} min", flush=True)
    q = list(todo); lock = threading.Lock(); stop = threading.Event(); errs = [0]; done = [0]

    def worker():
        while True:
            with lock:
                if not q or stop.is_set(): return
                if time.time() > a.deadline - START_MARGIN:
                    print("[run_legs] start margin reached: no new tries", flush=True); q.clear(); return
                i = q.pop(0)
            if a.kind == "pelican":
                prompt, seed, mt, tag, ref = PROMPT, i, a.max_tokens or PEL_MAX_TOK, f"{i:02d}", None
            else:
                prompt, seed, mt, tag, ref = rows[i]["problem"] + MATH_SUFFIX, 1000 + i, a.max_tokens or MATH_MAX_TOK, f"{i:03d}", rows[i]["answer"]
            try:
                r = run_try(base, a.kind, tag, prompt, seed, mt, outdir, a.arm, stop, ref)
            except Exception as e:
                errs[0] += 1; print(f"[run_legs] ERROR item {i}: {e}", flush=True)
                if errs[0] > 3: stop.set()
                continue
            if r is None: return
            done[0] += 1
            print(f"[run_legs] {a.arm} {a.kind}-{tag} fin={r['finish_reason']} think={r['think_tokens']} ans={r['answer_tokens']} complete={r.get('complete')} wall={r['wall_s']}s tps={r.get('decode_tps_server')}", flush=True)

    def watchdog():
        while not stop.is_set():
            if time.time() > a.deadline - STOP_MARGIN:
                print("[run_legs] stop margin: aborting in-flight tries (not saved)", flush=True); stop.set(); return
            time.sleep(5)
    wd = threading.Thread(target=watchdog, daemon=True); wd.start()
    ts = [threading.Thread(target=worker) for _ in range(a.slots)]
    [t.start() for t in ts]; [t.join() for t in ts]
    stop.set()
    left = [i for i in items if not os.path.exists(os.path.join(outdir, f"{a.kind}-{i:02d}.json" if a.kind == "pelican" else f"{a.kind}-{i:03d}.json"))]
    print(f"[run_legs] leg end: done now {done[0]}, items without a result {len(left)}, errors {errs[0]}", flush=True)
    sys.exit(0 if not left else (1 if errs[0] > 3 else 3))


if __name__ == "__main__":
    main()
