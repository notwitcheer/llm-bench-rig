"""Tier-2: reproduce the QuTLASS MXFP4 4x-on-5090 regime — HF Transformers, MXFP4 (MR-GPTQ) vs
bf16, batch sweep (their 4x peaks at large batch). Decode throughput; warm-up discarded.

Run (capsule): PYTHONPATH=$PWD HF_HUB_OFFLINE=1 ~/qutlass-env/bin/python scripts/bench_qutlass.py \
  --model ~/models/<mxfp4-ckpt> --arm mxfp4 --batches 1,8,32
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Explain, step by step, how a CPU executes a single machine instruction."


def bench(model, tok, batch, max_new):
    ids = tok([PROMPT] * batch, return_tensors="pt", padding=True).to(model.device)
    gen_kw = dict(max_new_tokens=max_new, do_sample=False, use_cache=True,
                  pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        model.generate(**ids, **gen_kw)                    # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**ids, **gen_kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    gen = (out.shape[1] - ids["input_ids"].shape[1]) * batch
    return gen / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)                # mxfp4 | bf16
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for b in [int(x) for x in a.batches.split(",")]:
        try:
            tps = bench(model, tok, b, a.max_new)
            rows.append({"batch": b, "tps": round(tps, 1)})
            print(f"{a.arm} batch={b}: {tps:.1f} tok/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": b, "tps": None, "oom": True})
            print(f"{a.arm} batch={b}: OOM", flush=True)
            break

    os.makedirs("results/fp4", exist_ok=True)
    rec = {"arm": a.arm, "model": a.model, "batches": rows,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    json.dump(rec, open(f"results/fp4/qutlass__{a.arm}.json", "w"))
    print(f"-> results/fp4/qutlass__{a.arm}.json", flush=True)


if __name__ == "__main__":
    main()
