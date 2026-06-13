#!/usr/bin/env python3
"""Single-stream decode tok/s + acceptance metrics for ONE spec-decode config on a
running vLLM endpoint. The three-way (MTP vs EAGLE3 vs DFlash) is built by running this
once per config (baseline + 3 drafters), same target model, then charting the set.

PURE MEASUREMENT CLIENT by design (RUNBOOK topology): the operator launches/swaps the
heavy vLLM server in tmux (Donald stopped first), this script only does HTTP — so it
never manages VRAM and never trips the SSH pkill/&->255 gotchas.

  capsule$ vllm serve google/gemma-4-26B-A4B-it --quantization compressed-tensors \\
             --speculative-config '{"method":"dflash","model":"z-lab/gemma-4-26B-A4B-it-DFlash",
                                    "num_speculative_tokens":15,"attention_backend":"flash_attn"}' &
  capsule$ .venv/bin/python scripts/bench_specdecode.py --base http://127.0.0.1:8000/v1 \\
             --model google/gemma-4-26B-A4B-it --target gemma-4-26b-a4b --label dflash \\
             --method dflash --num-spec 15 --out results/gemma-4-26b-a4b/specdecode-dflash.json

Workload prompts mirror bench_mtp_workload.sh so vLLM numbers compare to our published
llama.cpp MTP baselines. Acceptance is read from vLLM /metrics as a delta over the run
window (cumulative counters), so baseline (no spec) cleanly yields null acceptance.
"""
import argparse
import json
import os
import statistics
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.speed_vllm import _root, measure_chat_completion, prefill_decode_tps  # noqa: E402
from lib.specdecode import delta, parse_spec_metrics, summarize_acceptance  # noqa: E402

# Same five as bench_mtp_workload.sh — the workload spread is the whole point (MTP/EAGLE
# acceptance moves with output predictability; DFlash claims to stay flat across it).
WORKLOADS = {
    "prose": "Write a thorough, detailed explanation (at least 150 words) of the tradeoffs of "
             "running large language models locally on a single consumer GPU: memory capacity, "
             "bandwidth, quantisation, and context length.",
    "Q&A": "Explain in detail, step by step, what a KV cache is in a transformer, why it grows "
           "with context length, and how it affects both inference speed and memory use.",
    "JSON": "Output a JSON array of 30 user objects, each with fields id, name, email, score, "
            "active. Output only the JSON.\n[",
    "code": "Write a thread-safe LRU cache in Python with get and put methods, full docstrings, "
            "type hints, and a short usage example.\n```python\n",
    "repetitive": "Output 60 sequential log lines, each exactly in the form STATUS: OK seq=<n> "
                  "with n incrementing from 1.\nSTATUS: OK seq=1\n",
}


def fetch_metrics(base_url: str) -> dict[str, float]:
    """Scrape spec-decode counters from vLLM /metrics (root, not /v1). Empty on baseline."""
    url = f"{_root(base_url)}/metrics"
    try:
        with httpx.Client(timeout=15) as c:
            return parse_spec_metrics(c.get(url).text)
    except Exception as e:  # /metrics absent or server hiccup -> acceptance simply unknown
        print(f"[specdecode] WARN: /metrics unavailable ({e}); acceptance will be null", file=sys.stderr)
        return {}


def run(base, model, n_predict, reps=1):
    before = fetch_metrics(base)
    workloads = {}
    for name, prompt in WORKLOADS.items():
        samples, out_toks = [], 0
        for _ in range(reps):
            m = measure_chat_completion(base, model, prompt, max_tokens=n_predict)
            tps = prefill_decode_tps(m["prompt_tokens"], m["ttft_s"], m["output_tokens"], m["total_s"])
            samples.append(tps["decode_tps"])
            out_toks = m["output_tokens"]
        med = round(statistics.median(samples), 2)
        workloads[name] = {"decode_tps": med, "output_tokens": out_toks, "reps": reps, "samples": samples}
        print(f"[specdecode] {name:11s} {med:7.1f} tok/s (median of {reps}, {out_toks} tok)", file=sys.stderr)
    after = fetch_metrics(base)
    return workloads, summarize_acceptance(delta(before, after))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="vLLM OpenAI base, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, help="model slug for the results dir, e.g. gemma-4-26b-a4b")
    ap.add_argument("--label", required=True, help="baseline | mtp | eagle3 | dflash")
    ap.add_argument("--method", default=None, help="vLLM speculative method (null for baseline)")
    ap.add_argument("--num-spec", type=int, default=None, dest="num_spec")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--reps", type=int, default=1, help="runs per workload; reports the median (kills single-run noise)")
    ap.add_argument("--think", action="store_true", help="record think-on (ADR-0002); default off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    workloads, acceptance = run(a.base, a.model, a.n_predict, a.reps)
    result = {
        # ADR-0002 spirit: every result is think-labeled, even speed-only ones.
        "meta": {
            "label": a.label, "method": a.method, "num_spec": a.num_spec,
            "model": a.model, "target": a.target, "backend": "vLLM", "think": a.think,
        },
        "workloads": workloads,
        "acceptance": acceptance,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    al = acceptance.get("acceptance_length")
    print(f"[specdecode] {a.label}: acceptance_length={al} -> wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
