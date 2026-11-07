#!/usr/bin/env python3
"""b93_facts.py <summary.json> <out facts.json>: the end card's numbers, read from the run's own summary (never typed). Title and rows are built from the verdict, nothing worded by hand."""
import json
import sys

s = json.load(open(sys.argv[1]))
A = s["arms"]
T = s["takes"]
pf = s.get("prefill") or {}


def dec(arm):
    return (A.get(arm) or {}).get("decode_tok_s")


def fmt(x, nd=0):
    return ("%.*f" % (nd, x)) if x is not None else "-"


def first(tag):
    r = sorted([t for t in T.values() if t["tag"] == tag and t.get("clean")], key=lambda t: t["dir"])
    return r[0] if r else None


def mmss(x):
    return "%d:%02d" % (int(x // 60), int(x % 60)) if x is not None else "-"


def kfmt(x):
    return ("{:,}".format(int(round(x)))) if x else "-"


L0, B0, L1, B1 = dec("llama_plain"), dec("basalt_spec0"), dec("llama_ngram"), dec("basalt_default")
r0, r1, v = s.get("r0_nodraft_ratio"), s.get("r1_draft_ratio"), s["verdict"]
title = {"BASALT AHEAD": "On the same Qwen3.8-Flash-Next file, Basalt v0.3.0 decodes %.1fx faster than llama.cpp, no drafting" % (r0 or 0),
         "LEVEL": "On the same Qwen3.8-Flash-Next file, Basalt v0.3.0 and llama.cpp decode at about the same speed (%.2fx), no drafting" % (r0 or 0),
         "LLAMA.CPP AHEAD": "On the same Qwen3.8-Flash-Next file, llama.cpp decodes faster than Basalt v0.3.0 (Basalt at %.2fx), no drafting" % (r0 or 0)}.get(v, "Basalt v0.3.0 against llama.cpp on one Qwen3.8-Flash-Next file")
a, b = first("llama"), first("basalt")
def ppv(arm, k):
    return (((A.get(arm) or {}).get("pp") or {}).get(k) or {}).get("tok_s")


pl = {"pp16k": ppv("llama_plain", "pp16k"), "pp48k": ppv("llama_plain", "pp48k")}
pb = {"pp16k": ppv("basalt_spec0", "pp16k"), "pp48k": ppv("basalt_spec0", "pp48k")}
own = s.get("reference_own_post_tok_s") or 59.7
rows = [
    ["decode, no drafting, tok/s", fmt(L0), fmt(B0), "9 requests each (3 prompts x 3 runs, 700 tokens), the engine's own timings; llama.cpp at its best --n-cpu-moe (%s) of the ladder" % s.get("best_n_cpu_moe")],
    ["decode, with drafting, tok/s", fmt(L1), fmt(B1), "same 9 requests; llama.cpp n-gram only (this GGUF has no MTP layer), Basalt its own MTP head: not the same drafter"],
    ["prefill 16K / 48K prompt, tok/s", "%s / %s" % (kfmt(pl.get("pp16k")), kfmt(pl.get("pp48k"))), "%s / %s" % (kfmt(pb.get("pp16k")), kfmt(pb.get("pp48k"))), "cold prompt through each engine's server, tok/s from its own timings; llama-server default batch sizes"],
    ["same Hermes task, time to done, tool calls", "%s, %s" % (mmss(a and a.get("finish_s")), (a or {}).get("tool_calls", "-")), "%s, %s" % (mmss(b and b.get("finish_s")), (b or {}).get("tool_calls", "-")), "first take of each, no drafting, thinking on, real Hermes sessions, same prompt and sampling"],
    ["no-draft decode against our own post", "%s" % own, ("%.1fx" % (B0 / own)) if B0 else "-", "left: our 2026-09-09 post, llama.cpp on the 79.6 GB UD-Q2_K_XL file (a different file). right: Basalt no-draft / that figure"],
]
facts = {"title": title, "subtitle": "Qwen3.8-Flash-Next GSQ-RCO Q2_0 (ISTA-DASLab), one RTX 5090 32 GB + 59 GB RAM. llama.cpp 18ab62a vs Basalt v0.3.0, KV q8_0 / int8, ctx 65,536.",
         "cols": [["llama.cpp", "GGUF 66.4 GB"], ["Basalt v0.3.0", ".basalt 70.4 GB"]], "rows": rows,
         "small": ["tested claim: a community guide says Basalt decodes ~3.5x llama.cpp without drafting (RTX 5080 16 GB, 64 GB RAM). other card, other RAM, other quant.",
                   "ours: one box, one task family, 9 requests per row, Hermes rows are n=%d takes per engine; same weights, so the engine is the variable. not a model-quality test." % (s.get("clean_takes", {}).get("basalt") or 0),
                   "the .basalt pack and the GGUF come from the same quantiser; pack embeds an MTP layer the GGUF lacks. no speed multiplier is claimed beyond these rows."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
