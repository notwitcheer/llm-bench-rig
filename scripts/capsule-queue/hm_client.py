#!/usr/bin/env python3
"""Humming vs Marlin bench client over an OpenAI /v1 endpoint (streaming, T=0, thinking off).
usage: hm_client.py <port> <leg> <out.json>
phase c1: 8 prompts one at a time, max_tokens 512, natural stop (texts kept for the outputs-match check).
phase c8: the same 8 prompts fired together, 2 rounds, max_tokens 512, ignore_eos (fixed length so total TPS compares cleanly).
perceived TPS = (completion_tokens - 1) / (t_last_token - t_first_token) per request; total TPS = sum tokens / round wall."""
import json, sys, os, time, urllib.request, statistics, hashlib, threading
MAXTOK = int(os.environ.get("HM_MAXTOK", "512"))

PORT, LEG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}/v1"
MODEL = json.load(urllib.request.urlopen(BASE + "/models", timeout=30))["data"][0]["id"]
PROMPTS = [
    "Explain how a hash table works, including collisions and resizing.",
    "Write a Python function that parses a CSV file and returns the average of a numeric column, with error handling. Then explain it.",
    "A train leaves at 09:40 and travels 237 km at 79 km/h. Another leaves the same station at 10:05 at 95 km/h on the same track. When and where does the second catch the first? Show your working.",
    "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
    "Summarise the main causes of the French Revolution in a structured answer with headings.",
    "Write a bash script that backs up a directory to a dated tar.gz, keeps the last 7 backups and logs each run. Explain each step.",
    "What are the differences between TCP and UDP? Give three real-world examples where each is the better choice.",
    "Translate the following into French and then explain two grammar points in the translation: 'I would have come earlier if the meeting had not been moved to Thursday.'",
]


def run(prompt, max_tokens=None, ignore_eos=False):
    max_tokens = max_tokens or MAXTOK
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
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
                d = ch.get("delta", {}).get("content") or ch.get("delta", {}).get("reasoning_content") or ""
                if d:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text.append(d)
    t_end = time.time()
    n = usage["completion_tokens"] if usage else None
    decode = (n - 1) / (t_last - t_first) if (n and t_last and t_last > t_first) else None
    txt = "".join(text)
    return {"ttft_s": round(t_first - t0, 4) if t_first else None, "total_s": round(t_end - t0, 3),
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": n, "decode_tok_s": round(decode, 2) if decode else None,
            "text": txt, "text_sha": hashlib.sha1(txt.encode()).hexdigest()[:12]}


def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 3)


for _ in range(2):
    run("Say hello in five words.", 32)
c1 = []
for i, p in enumerate(PROMPTS):
    r = run(p); r["prompt_idx"] = i; c1.append(r)
    print(LEG, "c1", i, r["completion_tokens"], r["decode_tok_s"], r["ttft_s"], flush=True)

c8_rows, rounds = [], []
for rd in range(2):
    res = [None] * len(PROMPTS)
    def worker(i):
        res[i] = run(PROMPTS[i], MAXTOK, ignore_eos=True)
    th = [threading.Thread(target=worker, args=(i,)) for i in range(len(PROMPTS))]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] or 0 for r in res)
    rounds.append({"round": rd, "wall_s": round(wall, 3), "tokens": toks, "total_tps": round(toks / wall, 2)})
    for i, r in enumerate(res):
        r["prompt_idx"] = i; r["round"] = rd; r.pop("text", None); c8_rows.append(r)
    print(LEG, "c8 round", rd, rounds[-1], flush=True)


def summ(rows):
    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    tt = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    return {"n": len(dec), "perceived_tps_p50": pct(dec, 0.5), "perceived_tps_p90": pct(dec, 0.9),
            "perceived_tps_min": min(dec), "perceived_tps_max": max(dec),
            "ttft_p50_s": pct(tt, 0.5), "ttft_p95_s": pct(tt, 0.95),
            "tokens": sum(r["completion_tokens"] or 0 for r in rows)}

s1 = summ(c1); s8 = summ(c8_rows)
s8["total_tps_rounds"] = [r["total_tps"] for r in rounds]
s8["total_tps_median"] = round(statistics.median(s8["total_tps_rounds"]), 2)
s1["total_tps"] = round(s1["tokens"] / sum(r["total_s"] for r in c1), 2)
summary = {"leg": LEG, "model": MODEL, "c1": s1, "c8": s8,
           "shape": "8 prompts (~30-60 prompt tokens), chat, enable_thinking false, T=0, max_tokens 512; c1 = one request at a time, natural stop; c8 = 8 in flight x 2 rounds, ignore_eos (512 out each); 2 warmups; prefix caching off"}
print(LEG, "SUMMARY", json.dumps(summary), flush=True)
json.dump({"summary": summary, "c1_rows": c1, "c8_rows": c8_rows, "c8_rounds": rounds}, open(OUT, "w"), indent=1)
