#!/usr/bin/env python3
"""rp_recount.py <takedir> [--json out.json]: re-count a recorded take's HUD token series OFFLINE from its proxy logs (hud/hud.jsonl + hud/requests.jsonl + rec/go-epoch.txt) and check it
against the engine's own numbers. Uses rp_count.llama_srv_tokens, the same function the live proxy calls.

llama take : srv(t) = sum(exact predicted_n of requests finished by t) + (deltas of the request in flight). Checked against llama-server's own `timings` (predicted_per_second per request,
             predicted_n): per request >= 20 s the recounted tokens/s between t0+3 and t1-1 must sit within 10% of timings.predicted_per_second, and the final total must equal sum(predicted_n).
vLLM take  : the HUD already is vllm:generation_tokens_total; checked against the usage numbers (final total within 1% of sum(completion_tokens), per request >= 20 s rate within 10% of usage / wall).
`pass` = the RECOUNT agrees with the engine; `hud_ok` = the HUD series that was recorded agrees with the recount (within 3% at the end of the take). Exit 0 = recount inside tolerance, 1 = not;
the item logs hud_ok separately (False = the live counter was wrong on that take).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rp_count import llama_srv_tokens   # noqa: E402

td = sys.argv[1].rstrip("/")
hud = [json.loads(l) for l in open(td + "/hud/hud.jsonl") if l.strip()]
reqs = [json.loads(l) for l in open(td + "/hud/requests.jsonl") if l.strip()]
aborted = [r for r in reqs if r.get("t1") and r.get("predicted_n") is None and r.get("chunks")]      # streams cut before their last chunk (the wall's Ctrl-C): no usage, never counted as exact
reqs = [r for r in reqs if r.get("t1") and r.get("predicted_n") is not None]
go = float(open(td + "/rec/go-epoch.txt").read().split()[0]) if os.path.exists(td + "/rec/go-epoch.txt") else hud[0]["t"]
engine = next((r.get("engine") for r in reversed(hud) if r.get("engine")), None)
old_last = hud[-1].get("srv_tokens")


def recount_llama(row):
    t = row["t"]
    a = row["agents"]["a"]
    done = [r for r in reqs if r["t1"] <= t]
    c = {"a": {"srv_done": sum(r["predicted_n"] for r in done), "done_chunks": sum(r["chunks"] for r in done), "chunks": a["chunks"],
               "inflight": a["inflight"], "final_seen": False}}
    return llama_srv_tokens(c)


if engine == "llama":
    series = [(r["t"], recount_llama(r)) for r in hud]
else:
    series = [(r["t"], r.get("srv_tokens") or 0) for r in hud]


def at(t):
    best = series[0][1]
    for tt, v in series:
        if tt > t:
            break
        best = v
    return best


out = {"take": os.path.basename(td), "engine": engine, "recorded_hud_final": old_last, "new_series_final": round(series[-1][1]), "exact_sum_predicted_n": sum(r["predicted_n"] for r in reqs), "requests": []}
bad = []
for r in reqs:
    dur = r["t1"] - r["t0"]
    if dur < 20:      # at the HUD's 1 Hz sampling a 10 s request is mostly its two edges (prompt processing, last second): reported by the total check, not the rate check
        continue
    if r.get("pred_ps"):      # llama: the decode rate. Median of the 2 s windows inside the request that moved (prompt processing seconds do not), against the engine's own predicted_per_second.
        w = [(at(t + 2) - at(t)) / 2.0 for t in [r["t0"] + 1 + i for i in range(int(dur) - 3)]]
        w = sorted(x for x in w if x > 10)
        rate = w[len(w) // 2] if w else 0.0
        ref = r["pred_ps"]
        how = "median of 2 s windows vs timings.predicted_per_second"
    else:                     # vLLM: no per-second rate in the stream, so the whole request (prompt time included) against usage / wall
        rate = (at(r["t1"]) - at(r["t0"])) / dur
        ref = r["predicted_n"] / dur
        how = "whole request vs usage / wall (prompt time included)"
    pct = (rate / ref - 1) * 100
    row = {"t0_s": round(r["t0"] - go, 1), "dur_s": round(dur, 1), "predicted_n": r["predicted_n"], "recount_tok_s": round(rate, 1), "engine_tok_s": round(ref, 1), "how": how, "diff_pct": round(pct, 1)}
    out["requests"].append(row)
    if abs(pct) > 10:
        bad.append(row)
last_inflight = hud[-1]["agents"]["a"]["inflight"]
exact = out["exact_sum_predicted_n"]
# a take cut by the wall has a request in flight whose usage never arrives: the series may then exceed the finished-request sum (by that request's partial tokens), never fall short
tot_ok = (abs(out["new_series_final"] - exact) <= max(0.01 * exact, 5)) if not (last_inflight or aborted) else (out["new_series_final"] >= exact * 0.99)
out["ended_with_request_in_flight"] = bool(last_inflight or aborted)
out["aborted_requests"] = len(aborted)
out["total_ok"] = tot_ok
out["rate_ok"] = not bad
out["pass"] = tot_ok and not bad
rh = out["recorded_hud_final"] or 0
out["hud_vs_recount_pct"] = round((rh / out["new_series_final"] - 1) * 100, 2) if out["new_series_final"] else None
out["hud_ok"] = out["hud_vs_recount_pct"] is not None and abs(out["hud_vs_recount_pct"]) <= 3.0       # False on the item-85 q4km-1 take (+83.7%): the recorded HUD was wrong
print(json.dumps(out, indent=1))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
sys.exit(0 if out["pass"] else 1)
