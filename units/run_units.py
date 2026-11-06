#!/usr/bin/env python3
"""units driver: single-turn tool-call requests to a llama-server /v1/chat/completions (started with --jinja), one request in flight, greedy.
resumable per (model, task, form): rows already in <out>/<model>.jsonl are skipped. results are raw (the model's message); score.py classifies them.
usage: run_units.py --model KEY --port N --out DIR [--models models.json] [--tasks t01,u01] [--deadline-epoch E] [--host 127.0.0.1]
exit: 0 all rows of this model present | 3 stopped at the deadline with rows remaining | 4 server unhealthy / repeated transport errors | 2 usage"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ucommon as C  # noqa: E402


def http_json(url, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--deadline-epoch", type=int, default=0)
    a = ap.parse_args()
    spec = C.load_tasks()
    mods = C.load_models(a.models)
    mcfg = next((m for m in mods["legs"] if m["key"] == a.model), None)
    if mcfg is None:
        print("unknown model key", a.model)
        return 2
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.model + ".jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done.add((r["task"], r["role"]))
    want = None if a.tasks == "all" else set(a.tasks.split(","))
    todo = [(t, f) for t in spec["tasks"] if want is None or t["id"] in want for f in t["forms"] if (t["id"], f["role"]) not in done]
    base = "http://%s:%d" % (a.host, a.port)
    try:
        h = http_json(base + "/health", timeout=10)
        if "ok" not in json.dumps(h).lower():
            print("server not healthy", h)
            return 4
    except Exception as e:
        print("health check failed", repr(e))
        return 4
    print("[units] model %s: %d done, %d to do" % (a.model, len(done), len(todo)), flush=True)
    errs = 0
    n_new = 0
    with open(path, "a") as out:
        for t, f in todo:
            if a.deadline_epoch and time.time() >= a.deadline_epoch:
                print("[units] deadline reached, %d rows written this call" % n_new, flush=True)
                return 3
            payload = C.build_payload(spec, t, f, mcfg)
            t0 = time.time()
            try:
                body = http_json(base + "/v1/chat/completions", payload)
            except Exception as e:
                errs += 1
                print("[units] transport error %s (%d in a row)" % (repr(e)[:120], errs), flush=True)
                if errs >= 4:
                    return 4
                time.sleep(2)
                continue
            errs = 0
            lat = round(time.time() - t0, 3)
            p = C.parse_message(body)
            row = {"model": a.model, "task": t["id"], "role": f["role"], "tool": t["tool"], "real": t["real"], "prompt_sha256": f["prompt_sha256"],
                   "payload_sha256": C.sha(json.dumps(payload, sort_keys=True)), "latency_s": lat, "usage": body.get("usage"), "finish_reason": p["finish_reason"],
                   "n_calls": p["n_calls"], "name": p["name"], "args_raw": p["args_raw"], "content": p["content"][:1500], "reasoning_chars": p["reasoning_chars"],
                   "parse_error": p["error"], "parser": C.PARSER_VERSION}
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_new += 1
    print("[units] model %s complete: %d new rows" % (a.model, n_new), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
