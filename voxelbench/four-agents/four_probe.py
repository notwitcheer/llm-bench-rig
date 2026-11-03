#!/usr/bin/env python3
"""four_probe.py <upstream> <model-alias> <out.json>: the speed baseline for item 80, on the SAME server the race uses.

solo: one non-streamed chat request (320 tokens max, thinking as the server default) on an idle server, twice, median tok/s from llama-server's own `timings`.
four: four of the same request at once (four slots busy), twice, per-stream tok/s from `timings` and an aggregate = sum(predicted_n) / wall of the batch.
Short and on purpose boring: it is a baseline for the HUD ('one agent alone: N tok/s'), not a benchmark. Stdlib only. Never part of the race.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

up, alias, out = sys.argv[1:4]
PROMPT = "Write a JavaScript function drawIsoCube(ctx, x, y, size, colours) that draws one isometric cube on a 2D canvas with three shaded faces, then show how to call it in a loop to draw a 5 by 5 island. Add short comments."


def one(res, i):
    body = json.dumps({"model": alias, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 320, "stream": False}).encode()
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(up.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=240)
        j = json.loads(r.read())
        tm = j.get("timings") or {}
        res[i] = {"predicted_n": tm.get("predicted_n") or (j.get("usage") or {}).get("completion_tokens"), "tps": tm.get("predicted_per_second"), "prompt_n": tm.get("prompt_n"),
                  "wall_s": round(time.time() - t0, 2)}
    except Exception as e:
        res[i] = {"error": str(e)[:120]}


def batch(n):
    res = [None] * n
    ths = [threading.Thread(target=one, args=(res, i)) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [r for r in res if r and r.get("tps")]
    tot = sum(r["predicted_n"] or 0 for r in ok)
    return {"streams": res, "wall_s": round(wall, 2), "each_tps": round(statistics.mean(r["tps"] for r in ok), 2) if ok else None, "agg_tps": round(tot / wall, 2) if ok else None}


warm = batch(1)                      # warm-up (kernels, prompt cache): discarded
solo = [batch(1) for _ in range(2)]
four = [batch(4) for _ in range(2)]
solo_tps = statistics.median(s["each_tps"] for s in solo if s["each_tps"]) if any(s["each_tps"] for s in solo) else None
four_each = statistics.mean(f["each_tps"] for f in four if f["each_tps"]) if any(f["each_tps"] for f in four) else None
four_agg = statistics.mean(f["agg_tps"] for f in four if f["agg_tps"]) if any(f["agg_tps"] for f in four) else None
json.dump({"solo_tps": solo_tps, "four_each_tps": four_each, "four_agg_tps": four_agg, "warmup": warm, "solo_runs": solo, "four_runs": four,
           "note": "320-token requests, thinking on (server default), timings from llama-server; aggregate = predicted tokens / batch wall time"}, open(out, "w"), indent=1)
print("solo", solo_tps, "four each", four_each, "four agg", four_agg)
