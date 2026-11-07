#!/usr/bin/env python3
"""generation driver: one model server on --port, cells = prompts x seeds, 4 in flight. rows are appended only for finished cells (resumable)."""
import argparse, json, os, sys, time, threading, urllib.request, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glbcommon as G
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--deadline-epoch", type=int, default=0)
ap.add_argument("--prompts", default="all"); ap.add_argument("--seeds", default="all"); ap.add_argument("--max-tokens", type=int, default=0)
ap.add_argument("--inflight", type=int, default=4); ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()
C = G.cfg(); SYS = G.system_prompt(); S = C["sampler"]
CTK = next((m.get("chat_template_kwargs") or {} for m in C["models"] if m["key"] == a.model), {})
pr = [p for p in G.prompts() if a.prompts == "all" or p["id"] in a.prompts.split(",")]
seeds = C["seeds"] if a.seeds == "all" else [int(x) for x in a.seeds.split(",")]
maxtok = a.max_tokens or C["max_tokens"]
os.makedirs(a.out + "/raw/" + a.model, exist_ok=True)
rows_path = f"{a.out}/{a.model}.jsonl"
done = set()
if os.path.exists(rows_path):
    for l in open(rows_path):
        if l.strip(): r = json.loads(l); done.add((r["prompt"], r["seed"]))
cells = [(sd, p) for sd in seeds for p in pr if (p["id"], sd) not in done]   # seed-major
print(f"[run] {a.model}: {len(cells)} cells to do ({len(done)} done), max_tokens {maxtok}, inflight {a.inflight}", flush=True)
lock = threading.Lock(); stop = threading.Event(); n_ok = [0]
def one(cell):
    sd, p = cell
    if stop.is_set(): return "skipped"
    rem = (a.deadline_epoch - time.time()) if a.deadline_epoch else 1e9
    if rem < 90: stop.set(); return "past-deadline"
    body = {"model": a.model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": p["text"]}], "seed": sd,
            "temperature": S["temperature"], "top_k": S["top_k"], "top_p": S["top_p"], "min_p": S["min_p"], "repeat_penalty": S["repeat_penalty"],
            "max_tokens": maxtok, "stream": False}
    if CTK: body["chat_template_kwargs"] = CTK
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://{a.host}:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(req, timeout=min(2400, max(60, rem - 30))))
    except Exception as e:
        print(f"[run] {a.model} {p['id']} s{sd} FAILED {type(e).__name__} {str(e)[:100]}", flush=True); return "error"
    el = time.time() - t0
    ch = j["choices"][0]; m = ch["message"]; u = j.get("usage", {}); tm = j.get("timings", {})
    json.dump({"content": m.get("content"), "reasoning_content": m.get("reasoning_content")}, open(f"{a.out}/raw/{a.model}/{p['id']}-s{sd}.json", "w"))
    row = {"model": a.model, "prompt": p["id"], "seed": sd, "finish_reason": ch.get("finish_reason"), "completion_tokens": u.get("completion_tokens"),
           "prompt_tokens": u.get("prompt_tokens"), "thinking_kwargs": CTK, "reasoning_chars": len(m.get("reasoning_content") or ""), "content_chars": len(m.get("content") or ""),
           "elapsed_s": round(el, 1), "predicted_per_second": tm.get("predicted_per_second"), "ts": int(time.time())}
    with lock:
        open(rows_path, "a").write(json.dumps(row) + "\n"); n_ok[0] += 1
    print(f"[run] {a.model} {p['id']} s{sd}: {row['completion_tokens']} tok, {row['finish_reason']}, {row['elapsed_s']} s ({row['predicted_per_second']} tok/s slot)", flush=True)
    return "ok"
with cf.ThreadPoolExecutor(a.inflight) as ex: res = list(ex.map(one, cells))
left = sum(1 for r in res if r != "ok")
print(f"[run] {a.model}: {n_ok[0]} new rows, {left} not done", flush=True)
sys.exit(0 if left == 0 else 3)
