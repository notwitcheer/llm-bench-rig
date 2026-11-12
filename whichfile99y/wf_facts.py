#!/usr/bin/env python3
"""wf_facts.py <summary.json> <out facts.json>: the 99y end card's numbers, read from the run's own summary (never typed).
Title and rows are built from the verdict; nothing is worded by hand. Derived from mc_facts.py (item 99a)."""
import json
import sys

s = json.load(open(sys.argv[1]))
T = s["takes"]
arms = s["arms"]
v = s["verdict"]
fb = s.get("file_bytes") or {}


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
    title = "unsloth's v3.1 re-cut made no difference on our Hermes task: same pass count on one 5090"
elif v in ("CHANGED", "NEW MORE PASSES", "NEW FEWER PASSES"):
    title = "unsloth's v3.1 re-cut changed our Hermes task: %s passes on one 5090" % (
        "more" if v.endswith("MORE PASSES") or (arms["new"]["pass"] > arms["old"]["pass"]) else "fewer")
elif v == "INCOMPLETE":
    title = "unsloth's v3.1 re-cut on our Hermes task (provisional: legs still running)"
else:
    title = "unsloth's v3.1 re-cut on our Hermes task (no clean comparison this run)"
ob, nb = fb.get("old"), fb.get("new")
rows = [
    ["takes that passed (5 of 5 tests)", "%s" % frac("old"), "%s" % frac("new"), "pass = pristine tests, tests/ untouched; the verdict counts these"],
    ["file size", gb(ob), gb(nb), "from the GGUF on disk, size-checked against the HF byte count"],
    ["decode, tok/s (server timings)", num((arms["old"] or {}).get("decode_tok_s")), num((arms["new"] or {}).get("decode_tok_s")), "llama-server's own timings, weighted by tokens; reported, not gating"],
    ["first take: time to done, tool calls", "%s, %s" % (mmss(first("old") and first("old").get("finish_s")), num(first("old") and first("old").get("tool_calls"))),
     "%s, %s" % (mmss(first("new") and first("new").get("finish_s")), num(first("new") and first("new").get("tool_calls"))), "clock from prompt submit, real Hermes runs"],
]
facts = {"title": title,
         "subtitle": "Qwen3.8-27B UD-Q4_K_M, old upload against unsloth's v3.1 file, same llama.cpp build, f16 KV, ctx 65,536, same Hermes task on one 32 GB card.",
         "cols": [["October file", "UD-Q4_K_M, 2025-10 upload"], ["v3.1 file", "UD-Q4_K_M, unsloth main 4af51b35"]],
         "rows": rows,
         "small": ["tested claim: the vendor says v3.1 is 5-10%% more accurate. our check: one file pair, one task (4 planted bugs), %d takes per arm. no claim beyond this task." % (
                       max(arms["old"].get("takes_run", 0), arms["new"].get("takes_run", 0))),
                   "same llama.cpp build, same flags, same sampling, same seed for both arms. the verdict is the pass count: a tie is a valid result.",
                   "pass counts are per take; a pass means the pristine tests passed, nothing else."]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
