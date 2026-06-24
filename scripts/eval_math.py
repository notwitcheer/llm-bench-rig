"""Greedy pass@1 generation per eval variant (vLLM, Blackwell-capable vllm-env). Writes
gens-only JSON; grading is a SEPARATE pass (scripts/grade_em.py under grader-env) so the
grader's pinned latex2sympy2/antlr deps never have to coexist with vLLM's.

Run (capsule): ~/vllm-env/bin/python scripts/eval_math.py \
  --model ~/models/Qwen2.5-Math-7B --variant paper --benchmark math500 --label base
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams

from lib.em import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=["paper", "fair", "format"])
    ap.add_argument("--benchmark", required=True)             # math500 | amc23
    ap.add_argument("--max-tokens", type=int, default=3072)   # fair -> 4096 (set by runner)
    ap.add_argument("--label", required=True)                 # base | em-step10 | em-step15
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(f"dataset/em/{a.benchmark}.jsonl") if l.strip()]
    prompts = [build_prompt(r["problem"], a.variant) for r in rows]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)   # greedy pass@1
    outs = llm.generate(prompts, sp)

    recs = []
    for r, o in zip(rows, outs):
        recs.append({"id": r["id"], "gold": str(r["answer"]),
                     "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)})
    os.makedirs("results/em", exist_ok=True)
    out = {"variant": a.variant, "benchmark": a.benchmark, "model": a.model, "label": a.label,
           "max_tokens": a.max_tokens, "n": len(rows), "gens": recs}
    path = f"results/em/{a.label}__{a.variant}__{a.benchmark}.gens.json"
    json.dump(out, open(path, "w"))
    print(f"generated {a.label}/{a.variant}/{a.benchmark}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
