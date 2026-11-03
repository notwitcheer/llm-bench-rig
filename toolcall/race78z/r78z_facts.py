#!/usr/bin/env python3
"""r78z_facts.py <results-dir> <race.mp4.summary.json> <out facts.json>

The end card's numbers, read from the item-78 result files themselves (never typed): <results-dir> = benchmark-rig/results/toolcall-v1 on capsule (nvfp4-vs-q4km-paired.json and the
two s5-on summary.json). Adds the finish times of the race clip's two takes. Exits 2 when the item-78 numbers are not the ones this item was staged against (85 vs 80, 56 vs 78): the
card must never quote a different run.
"""
import json
import os
import sys

res, race, out = sys.argv[1:4]
p = json.load(open(os.path.join(res, "nvfp4-vs-q4km-paired.json")))
s5 = p["pairs"]["s5-on: nvfp4 vs q4km"]
st = p["pairs"]["st-on: nvfp4 vs q4km"]
nv = json.load(open(os.path.join(res, "qwen3-8-27b-nvfp4-vllm-s5-on/summary.json")))["speed"]["decode_tok_s_median"]
q4 = json.load(open(os.path.join(res, "qwen3-8-27b-ud-q4-k-m-s5-on/summary.json")))["speed"]["decode_tok_s_median"]
if (s5["a_pass"], s5["b_pass"]) != (85, 80) or round(nv) != 56 or round(q4) != 78:
    print("item-78 numbers changed: s5 %s vs %s, decode %s vs %s" % (s5["a_pass"], s5["b_pass"], nv, q4))
    sys.exit(2)
r = json.load(open(race))


def mmss(t):
    return "%d:%02d" % (int(t // 60), int(t % 60))


def tk(side):
    x = r[side]
    return mmss(x["finish_s"]) + ("" if x["end"] == "done" else " (stopped)")


rows = [
    ["multi-turn tool calls (100 entries)", "%d of 100" % s5["a_pass"], "%d of 100" % s5["b_pass"], "a tie: %d vs %d entries differ, exact McNemar p=%s (rule fixed before the run)" % (s5["a_only"], s5["b_only"], s5["p_exact"])],
    ["single-turn tool calls (899 requests)", "%d" % st["a_pass"], "%d" % st["b_pass"], "p=%s, same direction as the first row: no gap" % st["p_exact"]],
    ["decode speed, median tok/s", "%d" % round(nv), "%d" % round(q4), "%.2f vs %.2f. vLLM stream count vs llama.cpp timings: two engines, two counters, read two numbers" % (nv, q4)],
    ["this Hermes task, one take each", tk("left"), tk("right"), "same prompt, same sampling, start synced: a picture, not a measurement"],
]
facts = {"title": "tool calls: a tie. decode speed: not.",
         "subtitle": "Qwen3.8-27B on one RTX 5090. unsloth NVFP4 on vLLM 0.32.0 against unsloth UD-Q4_K_M on llama.cpp b10371.",
         "rows": rows,
         "small": ["toolcall-v1 frozen slice (BFCL v4 raw mode), think on, one greedy run per file. NVFP4's run-to-run noise is unmeasured, so the 6 vs 1 lean is not a result.",
                   "this is the file plus the engine people run it on: engine, KV dtype and calibration all differ. vLLM --linear-backend cutlass, FP8 KV cache as shipped.",
                   "race: two takes, start synced (27.9 GB + 18 GB do not sit on one 32 GB card together), n=1 each. sources: queue item 78 (2026-11-03) and 78z."]}
json.dump(facts, open(out, "w"), indent=1)
print("facts ok: s5 %d vs %d p=%s, decode %.2f vs %.2f, race %s vs %s" % (s5["a_pass"], s5["b_pass"], s5["p_exact"], nv, q4, tk("left"), tk("right")))
