"""Single-stream greedy spec-decode bench (HF Transformers path). ONE verify loop
(lib.cacheback.spec_decode_loop); arms differ only by the drafter. Lossless by construction;
we assert it empirically vs the AR run in T6 and assert realized speedup <= MAT.
batch=1, greedy, warm-state (first prompt discarded from timing). GPU-only (capsule).

Run (capsule):
  ~/cacheback-env/bin/python scripts/bench_cacheback.py \
    --model ~/models/Qwen3-8B --arm pld --workload dataset/cacheback/code.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.cacheback import spec_decode_loop, mean_accepted_tokens, load_workload, pld_propose


def make_forward_argmax(model):
    @torch.inference_mode()
    def forward_argmax(seq, draft):
        cand = torch.tensor([seq + draft], device=model.device)
        logits = model(cand).logits[0]                       # [len(cand), vocab]
        base = len(seq) - 1                                  # position predicting the next token
        return logits[base: base + len(draft) + 1].argmax(-1).tolist()
    return forward_argmax


def make_propose(arm, max_draft):
    if arm == "ar":
        return lambda seq: []
    if arm == "pld":
        return lambda seq: pld_propose(seq, 1, max_draft)    # LL=1; FL via max_draft
    raise ValueError(f"arm '{arm}' not supported in Tier 1 (cacheback added in T7)")


def run(model, tok, rows, arm, max_new, max_draft):
    forward_argmax = make_forward_argmax(model)
    propose = make_propose(arm, max_draft)
    eos = tok.eos_token_id
    outs, advs, t0 = [], [], None
    for i, r in enumerate(rows):
        ids = tok(r["prompt"], return_tensors="pt").input_ids[0].tolist()
        if i == 1:                                           # warm: discard prompt 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        gen, a = spec_decode_loop(forward_argmax, ids, propose, max_new, eos)
        outs.append(gen)
        if i >= 1:
            advs.extend(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_tok = sum(len(o) for o in outs[1:])
    return outs, advs, n_tok / dt, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["ar", "pld", "cacheback"], required=True)
    ap.add_argument("--workload", required=True)              # path to jsonl
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-draft", type=int, default=3)       # paper FL=3
    a = ap.parse_args()

    rows = load_workload(a.workload)
    wl = rows[0]["workload"]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    outs, advs, tps, n_tok = run(model, tok, rows, a.arm, a.max_new, a.max_draft)
    mat = 1.0 if a.arm == "ar" else mean_accepted_tokens(advs)

    os.makedirs("results/cacheback", exist_ok=True)
    rec = {"arm": a.arm, "workload": wl, "model": a.model, "tok_per_s": tps, "mat": mat,
           "n_prompts": len(rows), "gen_tokens": n_tok, "max_new": a.max_new,
           "max_draft": a.max_draft,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
           "advances": advs, "output_ids": outs}
    out = f"results/cacheback/{a.arm}__{wl}.json"
    json.dump(rec, open(out, "w"))
    print(f"{a.arm}/{wl}: {tps:.1f} tok/s  MAT={mat:.3f}  "
          f"vram={rec['peak_vram_gb']:.1f}GB  -> {out}")


if __name__ == "__main__":
    main()
