#!/usr/bin/env python3
"""oai_bench.py: one streamed OpenAI-style request with the engine's own numbers, shared by items 93 (Basalt vs llama.cpp) and 94 (QSA vs stock). Stdlib only.
usage: oai_bench.py --port P --model ALIAS --corpus FILE --calib calib.json --tokens N --task prose|code|conv --max-tokens M --temp T [--seed S] [--cold] [--think off|on] --out result.json [--series series.jsonl]
* the prompt = [cold: an 8-hex session id line] + the first K characters of the corpus + a task line. K is calibrated ONCE per token budget against the first engine's /tokenize and stored in --calib,
  so both engines of an item read byte-identical text (the second engine only reads the file). Without /tokenize on the server the calib entry must already exist.
* `timings` of the last stream chunk (llama.cpp names; Basalt writes the same names) give prompt_n, prompt_ms, predicted_n, predicted_ms, predicted_per_second, draft_n, draft_n_accepted.
* series.jsonl (optional): one row per ~0.25 s: {t, processed, total, decoded}; `processed/total` come from llama-server's prompt_progress events when the server sends them (return_progress), else null.
Exit 0 when a final chunk with timings arrived."""
import argparse, json, os, random, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--tokens", type=int, default=0); ap.add_argument("--chars", type=int, default=0, help="prompt = the first N characters of the corpus (no tokenizer needed; both engines read identical text)"); ap.add_argument("--task", default="prose")
ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=None); ap.add_argument("--cold", action="store_true")
ap.add_argument("--min-p", type=float, default=0.05); ap.add_argument("--think", default="off"); ap.add_argument("--out", required=True); ap.add_argument("--series", default=None)
ap.add_argument("--progress", action="store_true", help="ask llama-server for prompt_progress events")
a = ap.parse_args()
TASKS = {
    "prose": "Write a detailed 300-word explanation, in your own words, of what the documentation above says about memory and skills.",
    "code": "Write a Python script using only the standard library that opens a server-sent-events stream from a URL and prints every event as one line of JSON, with a test file.",
    "essay": "Write an essay on the Odyssey. About 500 words.",
    "lru": "Write a Python class implementing an LRU cache with type hints and docstrings.",
    "oneline": "In one sentence, what is this document?",
    "conv": "Explain how a B-tree differs from an LSM tree, with a worked example of each. About 600 words.",
}
base = "http://127.0.0.1:%d" % a.port


def call(path, body=None, timeout=60):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


corpus = open(a.corpus, encoding="utf-8").read()
calib = json.load(open(a.calib)) if os.path.exists(a.calib) else {}
text = ""
if a.tokens:
    key = str(a.tokens)
    if key not in calib:
        lo, hi = 1000, min(len(corpus), a.tokens * 8)
        for _ in range(22):          # bisect the character count whose token count is the budget (the server's own tokenizer)
            mid = (lo + hi) // 2
            n = len(call("/tokenize", {"content": corpus[:mid]}, 120)["tokens"])
            lo, hi = (mid, hi) if n < a.tokens else (lo, mid)
            if hi - lo < 64:
                break
        calib[key] = lo
        json.dump(calib, open(a.calib, "w"))
    text = corpus[:calib[key]]
if a.chars:
    text = corpus[:a.chars]
sid = ("Session %08x\n\n" % random.getrandbits(32)) if a.cold else ""
prompt = (sid + text + "\n\n---\n" if text else sid) + TASKS[a.task]
body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": a.temp, "stream": True,
        "stream_options": {"include_usage": True}}
if a.temp > 0:
    body.update(top_p=0.95, top_k=20, min_p=a.min_p)
if a.seed is not None:
    body["seed"] = a.seed
if a.think == "off":
    body["chat_template_kwargs"] = {"enable_thinking": False}
if a.progress:
    body["return_progress"] = True
t0 = time.time()
ser = open(a.series, "w") if a.series else None
decoded, processed, total, first, last_row, tm, usage, fin = 0, None, None, None, 0.0, None, None, None
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
err = None
try:
    r = urllib.request.urlopen(req, timeout=3600)
    for raw in r:
        line = raw.decode("utf8", "replace").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        j = json.loads(p)
        now = time.time() - t0
        pp = j.get("prompt_progress")
        if pp:
            processed, total = pp.get("processed"), pp.get("total")
        ch = (j.get("choices") or [{}])
        ch = ch[0] if ch else {}
        d = ch.get("delta") or {}
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
            if first is None:
                first = now
            decoded += 1
        if ch.get("finish_reason"):
            fin = ch["finish_reason"]
        if j.get("timings"):
            tm = j["timings"]
        if j.get("usage"):
            usage = j["usage"]
        if ser and now - last_row >= 0.25:
            ser.write(json.dumps({"t": round(now, 2), "processed": processed, "total": total, "decoded": decoded}) + "\n")
            last_row = now
except Exception as e:
    err = str(e)[:200]
wall = time.time() - t0
if ser:
    ser.write(json.dumps({"t": round(wall, 2), "processed": processed, "total": total, "decoded": (tm or {}).get("predicted_n", decoded), "final": True}) + "\n")
    ser.close()
out = {"port": a.port, "model": a.model, "task": a.task, "tokens_asked": a.tokens, "chars": a.chars, "temp": a.temp, "seed": a.seed, "cold": a.cold, "max_tokens": a.max_tokens, "wall_s": round(wall, 2),
       "ttft_s": round(first, 2) if first is not None else None, "deltas": decoded, "finish_reason": fin, "usage": usage, "timings": tm, "error": err}
if tm:
    out.update(prompt_n=tm.get("prompt_n"), cache_n=tm.get("cache_n"), prompt_ms=tm.get("prompt_ms"), prefill_tok_s=tm.get("prompt_per_second"), predicted_n=tm.get("predicted_n"),
               decode_tok_s=tm.get("predicted_per_second"), draft_n=tm.get("draft_n"), draft_acc=tm.get("draft_n_accepted"))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out.get(k) for k in ("task", "tokens_asked", "prompt_n", "cache_n", "prefill_tok_s", "predicted_n", "decode_tok_s", "finish_reason", "wall_s", "error")}))
sys.exit(0 if tm and not err else 1)
