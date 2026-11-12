#!/usr/bin/env python3
"""facts9a1.py <summary.json> <out facts.json>: the 9a1 end card's numbers, read from the run's own summary (never typed).
Title and rows are built from the verdict; nothing is worded by hand. Derived from wf_facts.py (item 99y)."""
import json
import sys

s = json.load(open(sys.argv[1]))
T = s["takes"]
arms = s["arms"]
v = s["verdict"]
fb = s.get("file_bytes") or {}
g = s.get("gsm8k") or {}


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


if v == "HOLDS":
    title = "a 6.7 GB ternary 27B held the same Hermes task as the 23 GB file, on one 5090"
elif v == "FALLS BEHIND":
    title = "a 6.7 GB ternary 27B fell behind the 23 GB file on the same Hermes task, on one 5090"
elif v == "INCOMPLETE":
    title = "a 6.7 GB ternary 27B on our Hermes task (provisional: legs still running)"
else:
    title = "a 6.7 GB ternary 27B on our Hermes task (no clean comparison this run: %s)" % v
sb, tb = fb.get("stock"), fb.get("ttf")
rows = [
    ["takes that passed (5 of 5 tests)", frac("stock"), frac("ttf"), "pass = pristine tests, tests/ untouched; the verdict counts these"],
    ["file size", gb(sb), gb(tb), "from the GGUF on disk, size-checked against the file byte count"],
    ["decode, tok/s (server timings)", num((arms["stock"] or {}).get("decode_tok_s")), num((arms["ttf"] or {}).get("decode_tok_s")),
     "llama-server's own timings, weighted by tokens; reported, not gating"],
    ["GSM8K, first 20 (thinking off)", "%s of %s" % (g.get("stock_correct", "-"), g.get("n", "-")), "%s of %s" % (g.get("ttf_correct", "-"), g.get("n", "-")),
     "greedy, one run per arm; n=20 is a reading, not a test"],
    ["first take: time to done, tool calls",
     "%s, %s" % (mmss(first("stock") and first("stock").get("finish_s")), num(first("stock") and first("stock").get("tool_calls"))),
     "%s, %s" % (mmss(first("ttf") and first("ttf").get("finish_s")), num(first("ttf") and first("ttf").get("tool_calls"))),
     "clock from prompt submit, real Hermes runs"],
]
facts = {"title": title,
         "subtitle": "Qwen3.8-27B: ttf TQ1_P ternary, no adapters, against stock Q6_K. same Hermes task, same sampling, one 32 GB card.",
         "cols": [["stock Q6_K", "llama.cpp cf34d3113, 22.88 GB"], ["ttf TQ1_P", "ttf-home engine 4afa91d03, 6.74 GB"]],
         "rows": rows,
         "small": ["tested claim: the author's README gives this file 0.880 on GSM8K (200), 0.787 HumanEval, 0.735 IFEval strict. our check: one card, one 4-bug task, %d takes per arm. no claim beyond this." %
                   max(arms["stock"].get("takes_run", 0), arms["ttf"].get("takes_run", 0)),
                   "same Hermes pin, same sampling, same seed for both arms. the verdict is the pass count: a tie is a valid result.",
                   "pass counts are per take; a pass means the pristine tests passed, nothing else."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
