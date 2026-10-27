#!/usr/bin/env python3
"""hermes-settings fit + depth probe (stdlib only). One call = one running llama-server (one KV type x one ctx).

Measures what a Hermes local user sees when they change the KV cache type or the context size:
  - VRAM after load and after the deepest request (nvidia-smi, whole card, nothing else resident)
  - prefill tok/s and decode tok/s at depth (server `timings`, cache_prompt false, greedy)
  - a passkey retrieval at every depth (smoke level, NOT a score: 1 needle per request)

usage: hermes_settings_fitdepth.py --server http://127.0.0.1:8097 --label f16-c131072 --out cell.json
         --kv f16 --ctx 131072 --plan deep|vram
plan vram: one 256-token request (load + first request + VRAM). plan deep: 0 / 8k / 32k / 64k / 120k prompt tokens
(passkey at 50 % depth) plus a second 120k request with the passkey at 10 %.
The prompt is a seeded pseudo-log (hex ids), token counts are set with the server's own /tokenize.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import urllib.request

KEYWORDS = ["amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor"]


def http(server, path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0].strip()) if out else -1


def line(rng, i):
    return (f"log {i:06d}: unit {rng.getrandbits(32):08x} {KEYWORDS[rng.randrange(8)]} "
            f"reading {rng.randrange(100000)} status {rng.getrandbits(16):04x}")


def n_tokens(server, text):
    return len(http(server, "/tokenize", {"content": text, "add_special": False})["tokens"])


def build_prompt(server, target, frac, seed, key):
    """~target prompt tokens (user message body incl. question), passkey line at `frac` of the lines."""
    question = ("\n\nQuestion: what is the pass key hidden in the log above? Reply with the pass key digits only, "
                "then on a new line write the numbers 1 to 30 separated by single spaces.")
    needle = f"IMPORTANT: the pass key is {key}. Remember it."
    rng = random.Random(seed)
    sample = "\n".join(line(rng, i) for i in range(200))
    tpl = max(1.0, n_tokens(server, sample) / 200.0)
    n = max(2, int((target - 80) / tpl))
    for _ in range(4):
        rng = random.Random(seed)
        lines = [line(rng, i) for i in range(n)]
        lines.insert(int(len(lines) * frac), needle)
        text = "\n".join(lines) + question
        have = n_tokens(server, text)
        if abs(have - target) <= max(40, target // 200):
            break
        n = max(2, int(n * target / max(have, 1)))
    return text, have


def one(server, target, frac, seed):
    key = str(random.Random(seed + 7).randrange(10000, 99999))
    text, have = build_prompt(server, target, frac, seed, key) if target > 0 else (
        f"IMPORTANT: the pass key is {key}. Remember it.\n\nQuestion: what is the pass key? Reply with the pass key digits "
        "only, then on a new line write the numbers 1 to 30 separated by single spaces.", 0)
    body = {"messages": [{"role": "user", "content": text}], "temperature": 0.0, "seed": 0, "max_tokens": 96,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    v0 = vram_mib()
    t0 = time.time()
    r = http(server, "/v1/chat/completions", body)
    wall = time.time() - t0
    tm = r.get("timings", {})
    content = (r["choices"][0]["message"].get("content") or "")
    return {"target_tokens": target, "prompt_tokens_text": have, "prompt_tokens_server": r["usage"]["prompt_tokens"],
            "completion_tokens": r["usage"]["completion_tokens"], "needle_frac": frac, "key": key,
            "found": key in content, "answer_head": content[:60].replace("\n", " | "),
            "prefill_tok_s": round(tm.get("prompt_per_second", 0), 1), "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
            "prompt_ms": round(tm.get("prompt_ms", 0)), "wall_s": round(wall, 1),
            "vram_mib_before": v0, "vram_mib_after": vram_mib()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--plan", choices=["deep", "vram"], required=True)
    a = ap.parse_args()
    props = http(a.server, "/props", timeout=60)
    rec = {"label": a.label, "kv": a.kv, "ctx": a.ctx, "plan": a.plan, "build_info": props.get("build_info"),
           "total_slots": props.get("total_slots"), "n_ctx_reported": props.get("default_generation_settings", {}).get("n_ctx"),
           "vram_mib_loaded": vram_mib(), "requests": [], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    plan = [(256, 0.5)] if a.plan == "vram" else [(0, 0.5), (8192, 0.5), (32768, 0.5), (65536, 0.5), (120000, 0.5), (120000, 0.1)]
    for n, (target, frac) in enumerate(plan):
        if target + 200 > a.ctx:
            rec["requests"].append({"target_tokens": target, "skipped": "does not fit ctx"})
            continue
        try:
            rec["requests"].append(one(a.server, target, frac, 1000 + n))
        except Exception as ex:  # a failed deep request is data (OOM / ctx), keep going to the next one
            rec["requests"].append({"target_tokens": target, "error": repr(ex)[:300], "vram_mib_after": vram_mib()})
        print(json.dumps(rec["requests"][-1]), flush=True)
    rec["vram_mib_peak_seen"] = max([rec["vram_mib_loaded"]] + [q.get("vram_mib_after", 0) for q in rec["requests"]])
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
