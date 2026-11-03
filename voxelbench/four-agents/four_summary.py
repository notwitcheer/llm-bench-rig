#!/usr/bin/env python3
"""four_summary.py <rundir> <slug> <wall_s> <rung> <mtp> <hermes-version> <prompt-md5> > summary.json

One machine-readable verdict for the item: fit rows, speed probe, race numbers (from the renderer's summary), the four gate verdicts, per-agent request stats
(cache-hit share = how often a slot kept its agent's prefix), server-log trouble lines, the regime string, and a headline with the line-1 number filled in.
Prints JSON on stdout. A missing piece is reported as null, never invented.
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
out = {"slug": slug, "wall_s": int(wall_s), "rung": rung, "mtp": mtp, "hermes": hv, "prompt_md5": pmd5, "fit": jl(run + "/fit.jsonl"), "speed_probe": probe,
       "race": race, "gates": gates, "requests": reqs, "server_log_trouble_lines": trouble[:12], "server_log_trouble_count": len(trouble),
       "vram_peak_mib_sampled": max(vram) if vram else None, "n_tokens_max_any_slot": ntok_max, "mtp_draft_acceptance": spec,
       "mean_slots_busy_per_decode_last": last.get("busy_per_decode"), "seconds_with_all_four_slots_busy": busy4,
       "clip": {k: (os.path.getsize(run + "/out/" + k) if os.path.exists(run + "/out/" + k) else None) for k in ("four-agents-final.mp4", "race.mp4", "results.mp4", "contact.png")},
       "leakscan": js(run + "/leakscan.json"),
       "regime": "parallel 4 on one llama-server (pooled KV, rung %s), Qwen3.8-27B Q6_K q8_0 KV, MTP %s, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05, RB 16384, thinking on; plain first-turn prompt (not /goal), "
                 "toolsets terminal+file, approvals off, memory off, own HERMES_HOME per agent (not served by the main gateway); per-agent tok/s not comparable to single-slot series numbers" % (rung, mtp),
       "headline": {"line1": line1, "mean_agg_tps_3plus_active": mean_agg, "peak_agg_tps_5s": (race or {}).get("agg_peak_tps_5s"), "solo_tps": solo,
                    "agg_over_solo": round(mean_agg / solo, 2) if (mean_agg and solo) else None, "finish_s": fin, "gates_pass": "%d of 4" % npass,
                    "vram_peak_mib": (race or {}).get("vram_peak_mib")}}
print(json.dumps(out, indent=1))
