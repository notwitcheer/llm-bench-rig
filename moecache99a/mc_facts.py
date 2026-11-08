#!/usr/bin/env python3
"""mc_facts.py <summary.json> <out facts.json>: the end card's numbers, read from the run's own summary (never typed). Title and rows are built from the verdict, nothing worded by hand.
Derived from item 89's rp_facts.py."""
import json
import sys

s = json.load(open(sys.argv[1]))
T = s["takes"]
arms = s["arms"]
v = s["verdict"]
ratio = s.get("ratio_cache_over_plain")
tune = s.get("tune") or {}
probe = s.get("probe") or {}


def rows_of(tag):
    return [t for t in T.values() if t["tag"] == tag]


def first(tag):
    r = sorted(rows_of(tag), key=lambda t: t["dir"])
    return r[0] if r else None


def mmss(x):
    return "%d:%02d" % (int(x // 60), int(x % 60)) if x is not None else "-"


def frac(tag):
    f = arms.get(tag) or {}
    return "%d of %d" % (f.get("pass", 0), f.get("takes_run", 0)) if f.get("takes_run") else "not run"


def avg(tag, k):
    x = [t[k] for t in rows_of(tag) if t.get(k) is not None and t.get("clean")]
    return round(sum(x) / len(x), 1) if x else None


def num(x, suffix=""):
    return ("%s%s" % (x, suffix)) if x is not None else "-"


a, b = first("plain"), first("cache")
dp, dc = arms["plain"].get("decode_tok_s"), arms["cache"].get("decode_tok_s")
title = {"FASTER": "llama.cpp's new MoE cache made a 122B model %.2fx faster on one RTX 5090" % (ratio or 0),
         "SLOWER": "llama.cpp's new MoE cache made a 122B model slower on one RTX 5090 (%.2fx)" % (ratio or 0),
         "TIE": "llama.cpp's new MoE cache made no clear speed difference for a 122B model on one RTX 5090",
         "CACHE FAILED": "llama.cpp's new MoE cache did not run cleanly with a 122B model on one RTX 5090",
         "INCOMPLETE": "llama.cpp's new MoE cache on a 122B model, one RTX 5090 (provisional)"}.get(v, "llama.cpp's new MoE cache on a 122B model, one RTX 5090")
cfg = "cache %s MiB, share %s" % (tune.get("mib", "?"), tune.get("share", "?"))
if tune.get("source") == "fallback":
    cfg += " (fallback)"
rows = [
    ["decode speed, tok/s (server timings)", num(dp), num(dc), "llama-server's own timings, weighted by tokens"],
    ["fixed-prompt probe, tok/s (temp 0)", num((probe.get("plain") or {}).get("decode_tok_s")), num((probe.get("cache") or {}).get("decode_tok_s")), "same prompt, 256 tokens, no agent, 2nd run"],
    ["first take: time to done, tool calls", "%s, %s" % (mmss(a and a.get("finish_s")), num(a and a.get("tool_calls"))), "%s, %s" % (mmss(b and b.get("finish_s")), num(b and b.get("tool_calls"))), "clock from prompt submit, real Hermes runs"],
    ["takes at 5/5 tests, tests untouched", frac("plain"), frac("cache"), "reported only, not in the verdict"],
    ["GPU power W (mean) / VRAM MiB (peak)", "%s / %s" % (num(avg("plain", "gpu_w_mean")), num(max([t.get("vram_mib_max") or 0 for t in rows_of("plain")] or [None]) or None)),
     "%s / %s" % (num(avg("cache", "gpu_w_mean")), num(max([t.get("vram_mib_max") or 0 for t in rows_of("cache")] or [None]) or None)), "nvidia-smi at 1 Hz, whole card"],
]
facts = {"title": title, "subtitle": "Hermes Agent driving Ornith-1.5-122B (UD-IQ3_XXS, 46 GB) on a 32 GB card with 59 GB RAM. llama.cpp master of 8 Nov 2026, KV q8_0, ctx 65,536.",
         "cols": [["no cache", "--n-cpu-moe 24"], ["MoE cache", cfg]], "rows": rows,
         "small": ["tested claim: the cache PR's own tuner example (a 92 GB card, a different model) predicts a large gain. other box, model and file.",
                   "ours: one 5090, one file, n=%d + %d Hermes takes, pass line is a 1.10x / 0.90x band on token-weighted decode. no claim for other cards or models." % (arms["plain"].get("takes_run", 0), arms["cache"].get("takes_run", 0)),
                   "same model file and build for both arms. sampling per the Ornith model card (temp 0.6, top-k 20, top-p 0.95), seed 721 is ours."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
