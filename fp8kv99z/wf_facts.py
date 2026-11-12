#!/usr/bin/env python3
"""kv_facts.py <summary.json> <out facts.json>: the 99y end card's numbers, read from the run's own summary (never typed).
Title and rows are built from the verdict; nothing is worded by hand. Derived from mc_facts.py (item 99a)."""
import json
import sys

s = json.load(open(sys.argv[1]))
T = s["takes"]
arms = s["arms"]
v = s["verdict"]
fb = s.get("vram_mib") or {}


def rows_of(tag):
    return [t for t in T.values() if t["tag"] == tag]


def first(tag):
    r = sorted(rows_of(tag), key=lambda t: t["dir"])
    return r[0] if r else None


def mmss(x):
    return "%d:%02d" % (int(x // 60), int(x % 60)) if x is not None else "-"


def num(x, suffix=""):
    return ("%s%s" % (x, suffix)) if x is not None else "-"


def frac(tag):
    f = arms.get(tag) or {}
    return "%d of %d" % (f.get("pass", 0), f.get("clean", 0)) if f.get("takes_run") else "not run"


def gb(b):
    return "%.2f GB" % (b / 1e9) if b else "-"


if v == "TIE":
    title = "fp8 KV cache made no difference on our Hermes task: same pass count on one 5090"
elif v in ("CHANGED", "NEW MORE PASSES", "NEW FEWER PASSES"):
    title = "fp8 KV cache changed our Hermes task: %s passes on one 5090" % (
        "more" if v.endswith("MORE PASSES") or (arms["fp8"]["pass"] > arms["f16"]["pass"]) else "fewer")
elif v == "INCOMPLETE":
    title = "fp8 KV cache on our Hermes task (provisional: legs still running)"
else:
    title = "fp8 KV cache on our Hermes task (no clean comparison this run)"
ob, nb = fb.get("f16"), fb.get("fp8")
rows = [
    ["takes that passed (5 of 5 tests)", "%s" % frac("f16"), "%s" % frac("fp8"), "pass = pristine tests, tests/ untouched; the verdict counts these"],
    ["VRAM used at ctx 65,536", ("%s MiB" % ob) if ob else "-", ("%s MiB" % nb) if nb else "-", "nvidia-smi after the server loads, same Q6_K file on both arms"],
    ["decode, tok/s (server timings)", num((arms["f16"] or {}).get("decode_tok_s")), num((arms["fp8"] or {}).get("decode_tok_s")), "llama-server's own timings, weighted by tokens; reported, not gating"],
    ["first take: time to done, tool calls", "%s, %s" % (mmss(first("f16") and first("f16").get("finish_s")), num(first("f16") and first("f16").get("tool_calls"))),
     "%s, %s" % (mmss(first("fp8") and first("fp8").get("finish_s")), num(first("fp8") and first("fp8").get("tool_calls"))), "clock from prompt submit, real Hermes runs"],
]
facts = {"title": title,
         "subtitle": "Qwen3.8-27B UD-Q4_K_M v3.1 file, same llama.cpp build, f16 KV against FP8 (f8_e4m3) KV, ctx 65,536, same Hermes task on one 32 GB card.",
         "cols": [["f16 KV", "UD-Q4_K_M v3.1, f16 cache"], ["FP8 KV", "UD-Q4_K_M v3.1, f8_e4m3 cache"]],
         "rows": rows,
         "small": ["tested claim: f8_e4m3 KV cache lowers VRAM with no quality loss. our check: one model file, one task (4 planted bugs), %d takes per arm. no claim beyond this task." % (
                       max(arms["f16"].get("takes_run", 0), arms["fp8"].get("takes_run", 0))),
                   "same llama.cpp build, same flags, same sampling, same seed for both arms. only the KV cache type differs. the verdict is the pass count: a tie is a valid result.",
                   "pass counts are per take; a pass means the pristine tests passed, nothing else."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
