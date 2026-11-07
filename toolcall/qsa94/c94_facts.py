#!/usr/bin/env python3
"""c94_facts.py <summary.json> <out facts.json>: the end card's numbers, read from the run's own summary (never typed)."""
import json
import sys

s = json.load(open(sys.argv[1]))
A = s["arms"]
v = s["verdict"]
r = s.get("r255")


def d(arm, depth, k="decode_tok_s"):
    return ((A.get(arm) or {}).get("depths", {}).get(str(depth)) or {}).get(k)


def f0(x):
    return ("{:,}".format(int(round(x)))) if x else "-"


def mmss(x):
    return "%d:%02d" % (int(x // 60), int(x % 60)) if x is not None else "-"


title = {"QSA AHEAD": "With 255K tokens of context, the QSA build decodes %.1fx faster than the dense build on one RTX 5090" % (r or 0),
         "LEVEL": "With 255K tokens of context, QSA and the dense build decode at about the same speed (%.2fx) on one RTX 5090" % (r or 0),
         "STOCK AHEAD": "With 255K tokens of context, stock llama.cpp decodes faster than the QSA fork (QSA at %.2fx) on one RTX 5090" % (r or 0)}.get(v, "QSA against stock llama.cpp at 255K tokens on one RTX 5090")
ss = s.get("sessions") or {}
sq, sk = ss.get("qsa") or {}, ss.get("stock") or {}


def sc(x):
    z = x.get("score") or {}
    return "%s of %s" % (z.get("correct", "-"), z.get("of", "-"))


def decs(depth):
    return f0(d("stock", depth)), f0(d("qsa", depth))


rows = [
    ["decode at 255K tokens, tok/s", f0(d("stock", 255000)), f0(d("qsa", 255000)), "code task, 2 seeds, temp 1.0, cold prompt; the engine's own timings. author's claim for this card: 141.8 (%s here)" % (s.get("claim_check") or "-")],
    ["decode at 64K / 128K, tok/s", "%s / %s" % (f0(d("stock", 64000)), f0(d("stock", 128000))), "%s / %s" % (f0(d("qsa", 64000)), f0(d("qsa", 128000))), "same task and flags at each depth; empty context: %s (stock) / %s (QSA)" % (f0(d("stock", 0)), f0(d("qsa", 0)))],
    ["prefill of the 255K prompt, tok/s", f0(d("stock", 255000, "prefill_tok_s")), f0(d("qsa", 255000, "prefill_tok_s")), "cold request, prompt tokens over the engine's own prompt time"],
    ["Hermes Agent, 24 doc questions right", sc(sk), sc(sq), "real session, docs as prefill, no file tools; answer = the default value written in the docs, whole-number match"],
    ["peak VRAM of the session, MiB", f0(sk.get("vram_mib_max")), f0(sq.get("vram_mib_max")), "nvidia-smi, whole card"],
]
facts = {"title": title, "subtitle": "Qwen3.8-27B GSQ-RCO IQ3_S + DFlash2 drafter (4 drafts), ctx 262,144, q8_0 KV. fork pentaCoxian/Qwen-QSA-llama.cpp fe08b16 vs llama.cpp 18ab62a.",
         "cols": [["stock llama.cpp", "dense IQ3_S 12.1 GB"], ["QSA fork", "QSA IQ3_S 12.3 GB"]], "rows": rows,
         "small": ["tested claim: the fork's README (RTX 5090: 141.8 tok/s at 255K; stock not published for this card). a replay on our box, not the author's run.",
                   "ours: 2 seeds per depth, one task for the ratio, thinking off; the stock arm is the dense file of the same quantiser (QSA weights cannot run on stock).",
                   "the README reports ~7%% of sampled replies stopping within 16 tokens and two-hop recall 11/16 against 12/16: the author's numbers, not retested. early stops here: stock %s/%s, QSA %s/%s." % (
                       (A.get("stock") or {}).get("early_stop_probe", {}).get("under_16_tokens", "-"), (A.get("stock") or {}).get("early_stop_probe", {}).get("requests", "-"),
                       (A.get("qsa") or {}).get("early_stop_probe", {}).get("under_16_tokens", "-"), (A.get("qsa") or {}).get("early_stop_probe", {}).get("requests", "-"))]}
json.dump(facts, open(sys.argv[2], "w"), indent=1)
print("facts ok:", title)
