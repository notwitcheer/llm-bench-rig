"""Tie-aware losslessness + speedup-sanity guard for the cacheback spec-decode suite (GPU).

Greedy spec-decode is lossless BY CONSTRUCTION: greedy_accept emits a token only when it
equals the target model's own argmax. But verifying that against a SEPARATE pure-AR run is a
flawed oracle in bf16: at exact logit ties the argmax tie-break depends on CUDA reduction
order, which varies with the forward-pass tensor shape. A spec arm appends draft tokens
(different shape than AR), so a handful of ties resolve differently and then diverge by greedy
path-dependence. That is the model being genuinely indifferent, not a decoder defect.

So we verify losslessness MODULO TIES: every AR-vs-spec divergence must sit at a position
whose top1-top2 logit gap is within bf16 noise (<= EPS). A confident disagreement (large gap)
would be a real bug and aborts. Also asserts realized speedup <= MAT (the t036 bug class).

Run (capsule): PYTHONPATH=. HF_HUB_OFFLINE=1 python scripts/verify_cacheback_lossless.py \
    --model ~/models/Qwen3-8B --arms pld cacheback
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.cacheback import assert_speedup_sane, load_workload

EPS = 0.5  # one bf16 ULP near logit magnitude ~30 is ~0.12-0.25; a true tie is ~0.0


def first_div(a, s):
    for i in range(min(len(a), len(s))):
        if a[i] != s[i]:
            return i
    return (min(len(a), len(s))) if len(a) != len(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", default=["pld", "cacheback"])
    ap.add_argument("--workloads", nargs="+", default=["code", "chat", "copyctx"])
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def gap_at(prompt, ref_ids, idx):
        pids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        with torch.inference_mode():
            lg = model(torch.tensor([pids + ref_ids[:idx]], device=model.device)).logits[0, -1].float()
        v = torch.topk(lg, 2).values.tolist()
        return v[0] - v[1]

    worst, total_div, total_tok = 0.0, 0, 0
    for wl in a.workloads:
        rows = load_workload(f"dataset/cacheback/{wl}.jsonl")
        ar = json.load(open(f"results/cacheback/ar__{wl}.json"))
        for arm in a.arms:
            spec = json.load(open(f"results/cacheback/{arm}__{wl}.json"))
            gaps = []
            for p, (ai, si) in enumerate(zip(ar["output_ids"], spec["output_ids"])):
                total_tok += len(ai)
                i = first_div(ai, si)
                if i is not None:
                    total_div += 1
                    g = gap_at(rows[p]["prompt"], ai, i)
                    gaps.append(g)
                    worst = max(worst, g)
                    assert g <= EPS, (f"REAL DIVERGENCE {wl}/{arm} p{p}@{i}: "
                                      f"logit gap {g:.4f} > {EPS} — not a tie, investigate")
            sp = spec["tok_per_s"] / ar["tok_per_s"]
            assert_speedup_sane(sp, spec["mat"])
            gtxt = f"  tie-gaps={[round(x, 3) for x in gaps]}" if gaps else ""
            print(f"{wl:8} {arm:10} {len(ar['output_ids']) - len(gaps)}/{len(ar['output_ids'])} "
                  f"byte-exact, {len(gaps)} tie-flip  speedup={sp:.3f}x MAT={spec['mat']:.3f}{gtxt}")
    print(f"\nALL GUARDS PASS: {total_tok - total_div}/{total_tok} tokens byte-identical to AR; "
          f"{total_div} divergences ALL at bf16 ties (worst gap {worst:.4f} <= {EPS}).")


if __name__ == "__main__":
    main()
