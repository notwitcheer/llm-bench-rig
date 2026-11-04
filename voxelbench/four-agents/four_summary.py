#!/usr/bin/env python3
"""four_summary.py <rundir> <slug> <wall_s> <rung> <mtp> <hermes-version> <prompt-md5> > summary.json

One machine-readable verdict for the item: fit rows, speed probe, race numbers (from the renderer's summary), the four gate verdicts, per-agent request stats
(cache-hit share = how often a slot kept its agent's prefix), server-log trouble lines, the regime string, and a headline with the line-1 number filled in.
Prints JSON on stdout. A missing piece is reported as null, never invented.

80b additions (computed for any run, item 80 included): per-agent compaction events counted from each agent's own state.db (rows with _compressed_summary = 1, which
Hermes inserts once per compaction and soft-archives, never deletes), the home config version each agent ran with (state/<a>/home-version.txt), per-agent peak context
(request start and request end, from the server's own timings), an estimate of the shared pool's peak use, the proxy guard's holds, and the PRE-REGISTERED success rule:
  PASS = all four agents (1) saw no failed request (HTTP != 200, a dropped connection or an error event inside a stream) AND (2) ended on their own or on the wall
         ("done" / "wall", never "server_error" or an early Hermes exit) AND (3) the server log has no context-exceeded / no-memory-slot line AND (4) every agent has at
         least one completed request with server timings (a run with no data cannot pass).
  Anything else = FAIL (the verdict names the rule that broke). Gates, tok/s and compaction counts are reported next to it, never folded into it.
"""
import glob
import json
import os
import re
import sys

run, slug, wall_s, rung, mtp, hv, pmd5 = sys.argv[1:8]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def js(path):
    return json.load(open(path)) if os.path.exists(path) else None


race = js(run + "/out/race.mp4.summary.json")
probe = js(run + "/speed-probe.json")
gates = {}
for a in "abcd":
    g = js("%s/gate/%s/gate.json" % (run, a)) or {}
    gates[a] = {"pass": bool(g.get("pass")), "bytes": g.get("bytes"), "frame_diff": g.get("frame_diff"), "failed": [k for k, v in (g.get("checks") or {}).items() if not v],
                "no_html": g.get("no_html", False)}
reqs = {}
for r in jl(run + "/hud/requests.jsonl"):
    d = reqs.setdefault(r["agent"], {"requests": 0, "errors": 0, "prompt_tokens": 0, "cached_tokens": 0, "predicted_tokens": 0})
    d["requests"] += 1
    if r.get("status") != 200 or r.get("error"):
        d["errors"] += 1
    d["prompt_tokens"] += r.get("prompt_n") or 0
    d["cached_tokens"] += r.get("cache_n") or 0
    d["predicted_tokens"] += r.get("predicted_n") or 0
for a, d in reqs.items():
    tot = d["prompt_tokens"] + d["cached_tokens"]
    d["cache_hit_share"] = round(d["cached_tokens"] / tot, 3) if tot else None
trouble = []
for f in glob.glob(run + "/server-*.log"):
    for l in open(f, errors="replace"):
        if re.search(r"error|failed to|exceed|truncat|out of memory|memory slot", l, re.I) and "failed to load" not in l.lower() and "truncated = 0" not in l:
            trouble.append(os.path.basename(f) + ": " + l.strip()[:160])
hud = jl(run + "/hud/hud.jsonl")
vram = [h.get("vram_mib") for h in hud if h.get("vram_mib")]
busy4 = sum(1 for h in hud if (h.get("slots_busy") or 0) >= 4)
last = hud[-1] if hud else {}
spec = round(last["spec_accept"] / last["spec_draft"], 3) if last.get("spec_draft") else None
ntok_max = max((h.get("n_tokens_max") or 0 for h in hud), default=None)
mean_agg = (race or {}).get("agg_mean_tps_3plus_active")
solo = (probe or {}).get("solo_tps")
npass = sum(1 for g in gates.values() if g["pass"])
fin = (race or {}).get("finish_s") or {}
line1 = "one RTX 5090 is serving four Hermes Agents at once: same prompt, same local Qwen3.8-27B, four panes racing, %s tok/s combined on the card" % ("%d" % round(mean_agg) if mean_agg else "[N]")
import sqlite3


def compaction_events(path):
    """(summary rows, sessions' ineffective counter, message rows) from one agent's copied state.db; read-only, None when the copy is missing"""
    if not os.path.exists(path):
        return None
    try:
        c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        n = c.execute("select count(*) from messages where _compressed_summary = 1").fetchone()[0]
        inef = c.execute("select coalesce(max(compression_ineffective_count), 0) from sessions").fetchone()[0]
        rows = c.execute("select count(*) from messages").fetchone()[0]
        c.close()
        return {"compaction_events": n, "ineffective_counter": inef, "message_rows": rows}
    except Exception as e:
        return {"error": str(e)[:100]}


per_agent = {}
reqrows = jl(run + "/hud/requests.jsonl")
t_first = min((r["t0"] for r in reqrows), default=0)
for a in "abcd":
    rs = [r for r in reqrows if r["agent"] == a]
    ok = [r for r in rs if r.get("status") == 200 and not r.get("error") and r.get("prompt_n") is not None]
    start_ctx = [(r.get("cache_n") or 0) + (r.get("prompt_n") or 0) for r in ok]
    end_ctx = [s_ + (r.get("predicted_n") or 0) for s_, r in zip(start_ctx, ok)]
    res = js("%s/rec/result-%s.json" % (run, a)) or {}
    ver = None
    try:
        ver = open("%s/state/%s/home-version.txt" % (run, a)).read().strip()
    except OSError:
        pass
    d = {"home_version": ver, "end": res.get("end"), "requests": len(rs), "failed_requests": sum(1 for r in rs if r.get("status") != 200 or r.get("error")),
         "peak_ctx_request_start": max(start_ctx, default=None), "peak_ctx_request_end": max(end_ctx, default=None),
         "largest_single_generation": max((r.get("predicted_n") or 0 for r in ok), default=None),
         "guard_held_requests": sum(1 for r in rs if (r.get("guard_wait_s") or 0) > 0.3), "guard_wait_s": round(sum(r.get("guard_wait_s") or 0 for r in rs), 1)}
    d.update(compaction_events("%s/state/%s/state.db" % (run, a)) or {"compaction_events": None})
    fin_a = fin.get(a) if isinstance(fin, dict) else None
    pred = (reqs.get(a) or {}).get("predicted_tokens")
    d["tok_s_end_to_end"] = round(pred / fin_a, 1) if (pred and fin_a) else None
    w = sum(r.get("predicted_n") or 0 for r in ok)
    d["tok_s_decode_mean"] = round(sum((r.get("pred_ps") or 0) * (r.get("predicted_n") or 0) for r in ok) / w, 1) if w else None
    per_agent[a] = d
# pool use: sum over agents of (context at request start ramping to request end, then held while idle), sampled at every request boundary
pts = sorted({r[k] for r in reqrows for k in ("t0", "t1") if r.get(k)})


def ctx_at(a, t):
    v = 0
    for r in reqrows:
        if r["agent"] != a or r.get("prompt_n") is None or r.get("status") != 200 or r.get("error"):
            continue
        s0 = (r.get("cache_n") or 0) + (r.get("prompt_n") or 0)
        e0 = s0 + (r.get("predicted_n") or 0)
        if r["t0"] <= t <= r["t1"]:
            return s0 + (e0 - s0) * (t - r["t0"]) / max(r["t1"] - r["t0"], 1e-3)
        if r["t1"] < t:
            v = e0
    return v


pool_peak = max((sum(ctx_at(a, t) for a in "abcd") for t in pts), default=None)
t_end_all = {}
server_text = ""
for f in glob.glob(run + "/server-*.log"):
    server_text += open(f, errors="replace").read()
exceeded_lines = len(re.findall(r"context size has been exceeded|failed to find a memory slot|exceeds the available context", server_text, re.I))
bad = []
for a, d in per_agent.items():
    if d["failed_requests"]:
        bad.append("agent %s: %d failed request(s)" % (a, d["failed_requests"]))
    if d["end"] not in ("done", "wall"):
        bad.append("agent %s: ended as %s" % (a, d["end"]))
for a, d in per_agent.items():
    if d["peak_ctx_request_end"] is None:
        bad.append("agent %s: no completed request with server timings (a run with no data cannot pass)" % a)
if exceeded_lines:
    bad.append("server log: %d context-exceeded / no-memory-slot line(s)" % exceeded_lines)
rule = {"text": "PASS = all four agents: no failed request, ended 'done' or 'wall', server log free of context-exceeded lines, timings present for every agent", "pass": not bad, "broken": bad}
guard = None
hud_last = hud[-1] if hud else {}
if hud_last.get("guard"):
    guard = hud_last["guard"]
compaction = {"total_events": sum((d.get("compaction_events") or 0) for d in per_agent.values()), "per_agent": {a: d.get("compaction_events") for a, d in per_agent.items()},
              "homes": sorted({d["home_version"] for d in per_agent.values() if d["home_version"]})}
out = {"slug": slug, "wall_s": int(wall_s), "rung": rung, "mtp": mtp, "hermes": hv, "prompt_md5": pmd5, "fit": jl(run + "/fit.jsonl"), "speed_probe": probe,
       "race": race, "gates": gates, "requests": reqs, "server_log_trouble_lines": trouble[:12], "server_log_trouble_count": len(trouble),
       "vram_peak_mib_sampled": max(vram) if vram else None, "n_tokens_max_any_slot": ntok_max, "mtp_draft_acceptance": spec,
       "mean_slots_busy_per_decode_last": last.get("busy_per_decode"), "seconds_with_all_four_slots_busy": busy4,
       "clip": {k: (os.path.getsize(run + "/out/" + k) if os.path.exists(run + "/out/" + k) else None) for k in ("four-agents-final.mp4", "race.mp4", "results.mp4", "contact.png")},
       "leakscan": js(run + "/leakscan.json"), "per_agent": per_agent, "pool_peak_est_tokens": round(pool_peak) if pool_peak else None, "pool_guard": guard,
       "server_exceeded_lines": exceeded_lines, "compaction": compaction, "success_rule": rule,
       "regime": "parallel 4 on one llama-server (pooled KV, rung %s), Qwen3.8-27B Q6_K q8_0 KV, MTP %s, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05, RB 16384, thinking on; plain first-turn prompt (not /goal), "
                 "toolsets terminal+file, approvals off, memory off, own HERMES_HOME per agent (not served by the main gateway); per-agent tok/s not comparable to single-slot series numbers" % (rung, mtp),
       "headline": {"line1": line1, "mean_agg_tps_3plus_active": mean_agg, "peak_agg_tps_5s": (race or {}).get("agg_peak_tps_5s"), "solo_tps": solo,
                    "agg_over_solo": round(mean_agg / solo, 2) if (mean_agg and solo) else None, "finish_s": fin, "gates_pass": "%d of 4" % npass,
                    "vram_peak_mib": (race or {}).get("vram_peak_mib"), "success_rule": "PASS" if rule["pass"] else "FAIL", "success_rule_broken": bad,
                    "compaction_events": compaction["total_events"], "pool_peak_est_tokens": round(pool_peak) if pool_peak else None}}
print(json.dumps(out, indent=1))
