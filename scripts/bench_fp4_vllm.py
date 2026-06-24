"""Tier-1 FP4 reality check: decode throughput per quant in vLLM on sm_120. Greedy, warm-up
discarded, batch-1 + a sweep. vLLM auto-detects the quant from the checkpoint; we record the
DECLARED quant here and the ACTUAL kernel from the captured init log (lib.fp4.parse_quant_kernel).

Run (capsule): VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=/usr/local/cuda-13.0 \
  FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
  ~/vllm-env/bin/python scripts/bench_fp4_vllm.py --model ~/models/Qwen3-14B-AWQ --label awq
"""
import argparse
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

PROMPT = "Explain, step by step, how a CPU executes a single machine instruction."


def bench_batch(llm, batch, max_new):
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=True)
    prompts = [PROMPT] * batch
    llm.generate(prompts, sp)                          # warm-up (discarded)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    return gen / dt, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)              # bf16 | awq | fp8 | nvfp4
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.88,
              max_model_len=4096, enforce_eager=False)
    declared = str(getattr(llm.llm_engine.model_config, "quantization", None))

    rows = []
    for b in [int(x) for x in a.batches.split(",")]:
        try:
            tps, gen = bench_batch(llm, b, a.max_new)
            rows.append({"batch": b, "tps": round(tps, 1), "gen_tokens": gen})
            print(f"{a.label} batch={b}: {tps:.1f} tok/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": b, "tps": None, "oom": True})
            print(f"{a.label} batch={b}: OOM", flush=True)
            break

    os.makedirs("results/fp4", exist_ok=True)
    rec = {"label": a.label, "model": a.model, "declared_quant": declared,
           "batches": rows, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    json.dump(rec, open(f"results/fp4/vllm__{a.label}.json", "w"))
    print(f"-> results/fp4/vllm__{a.label}.json  declared_quant={declared}  "
          f"vram={rec['peak_vram_gb']}GB", flush=True)


if __name__ == "__main__":
    main()
