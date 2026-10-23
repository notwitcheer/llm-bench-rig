#!/usr/bin/env python3
"""Prefill (large-M GEMM) probe: 8 distinct ~2k-token prompts one at a time, max_tokens 1, streaming; TTFT p50/p95.
usage: hm_prefill.py <port> <leg> <out.json>"""
import json, sys, time, urllib.request, random

PORT, LEG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}/v1"
MODEL = json.load(urllib.request.urlopen(BASE + "/models", timeout=30))["data"][0]["id"]
WORDS = ("river mountain ledger copper violet harbour engine lantern meadow quartz signal orchard compass "
         "granite falcon timber canvas ember glacier saffron thimble beacon cobalt willow anchor").split()


def prompt(seed, n_words=1500):
    rnd = random.Random(seed)
    body = " ".join(rnd.choice(WORDS) for _ in range(n_words))
    return f"Document {seed}:\n{body}\n\nIn one word, which word appears first in the document above?"


def run(p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 1, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if (ch.get("delta", {}).get("content") or "") and t_first is None:
                    t_first = time.time()
    t_end = time.time()
    return {"ttft_s": round((t_first or t_end) - t0, 4), "prompt_tokens": usage.get("prompt_tokens") if usage else None}


def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 4)

for s in (900, 901):
    run(prompt(s))
rows = []
for i in range(8):
    r = run(prompt(i)); r["idx"] = i; rows.append(r)
tt = [r["ttft_s"] for r in rows]
pt = [r["prompt_tokens"] for r in rows]
summary = {"leg": LEG, "n": len(rows), "prompt_tokens_median": sorted(pt)[len(pt) // 2], "ttft_p50_s": pct(tt, 0.5),
           "ttft_p95_s": pct(tt, 0.95), "prefill_tok_s_p50": round(sorted(pt)[len(pt) // 2] / pct(tt, 0.5), 1),
           "shape": "8 distinct ~2k-token prompts, one at a time, max_tokens 1, T=0, prefix caching off, 2 warmups"}
print(LEG, "PREFILL", json.dumps(summary), flush=True)
json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
