#!/usr/bin/env python3
"""rp_facts.py <summary.json> <out facts.json>: the end card's numbers, read from the run's own summary (never typed). Title and rows are built from the verdict, nothing worded by hand."""
import json
import sys

s = json.load(open(sys.argv[1]))
T = s["takes"]
fv = s["files"]


def rows_of(tag):
    return [t for t in T.values() if t["tag"] == tag]


def first(tag):
    r = sorted(rows_of(tag), key=lambda t: t["dir"])
    return r[0] if r else None


def mmss(x):
    return "%d:%02d" % (int(x // 60), int(x % 60)) if x is not None else "-"


def frac(tag):
    f = fv.get(tag) or {}
    return "%d of %d" % (f.get("pass", 0), f.get("takes_run", 0)) if f.get("takes_run") else "not run"


def avg(tag, k):
    v = [t[k] for t in rows_of(tag) if t.get(k) is not None and t.get("clean")]
    return round(sum(v) / len(v), 1) if v else None


def num(x, suffix=""):
    return ("%s%s" % (x, suffix)) if x is not None else "-"


a, b = first("iq3xxs"), first("iq2s")
v = s["verdict"]
n3 = (fv.get("iq3xxs") or {}).get("takes_run", 0)
k3 = (fv.get("iq3xxs") or {}).get("pass", 0)
title = {"REPRODUCED": "Ripple-65B Preview fixed the 4-bug repo on one RTX 5090, fully local",
         "PARTIAL": "Ripple-65B Preview on one RTX 5090: %d of %d takes fixed the 4-bug repo" % (k3, n3),
         "NOT REPRODUCED": "Ripple-65B Preview on one RTX 5090: none of %d takes fixed the 4-bug repo" % n3}.get(v, "Ripple-65B Preview on one RTX 5090")
rows = [
    ["takes at 5/5 tests, tests untouched", frac("iq3xxs"), frac("iq2s"), "pristine tests re-run by the gate on each take's final package; a take that edits tests/ fails"],
    ["first take: time to done, tool calls", "%s, %s" % (mmss(a and a.get("finish_s")), num(a and a.get("tool_calls"))), "%s, %s" % (mmss(b and b.get("finish_s")), num(b and b.get("tool_calls"))), "clock from the prompt submit; real Hermes sessions, same prompt and sampling"],
    ["decode speed, tok/s (server timings)", num(avg("iq3xxs", "decode_tok_s_engine")), num(avg("iq2s", "decode_tok_s_engine")), "llama-server's own predicted_per_second over the take, no speculative decoding"],
    ["GPU power W (mean) / VRAM MiB (peak)", "%s / %s" % (num(avg("iq3xxs", "gpu_w_mean")), num(max([t.get("vram_mib_max") or 0 for t in rows_of("iq3xxs")] or [None]) or None)), "%s / %s" % (num(avg("iq2s", "gpu_w_mean")), num(max([t.get("vram_mib_max") or 0 for t in rows_of("iq2s")] or [None]) or None)), "nvidia-smi at 1 Hz during the take, whole card"],
]
facts = {"title": title, "subtitle": "Hermes Agent driving a dense 65B model on a 32 GB card. community GGUF files, llama.cpp PR #31692 build (not merged), KV q8_0, ctx 65,536.",
         "cols": [["IQ3_XXS", "28.0 GB"], ["IQ2_S", "25.3 GB"]], "rows": rows,
         "small": ["tested claim: a self-reported GB10 post (NVFP4 + speculative decoding, 5/5 in 1m 36s, 14 tool calls). other box, file, decoder, repo.",
                   "ours is a same-shape task (4 planted bugs, small Python library), n=%d takes on the main file, one decode regime, no speed multiplier claimed." % n3,
                   "Preview release, light post-training. sampling temp 0.6, top-p 0.95, min-p 0.05, seed 721 is ours, not a vendor setting."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
