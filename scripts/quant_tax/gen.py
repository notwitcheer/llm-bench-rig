"""t090: generate against a running llama-server. Gens only; grading is separate.

Splitting generation from grading keeps a grader change from silently invalidating
an expensive GPU pass, and lets HumanEval's subprocess execution run off the GPU box
if that is ever wanted.

Run (capsule, server already up):
  PYTHONPATH=. ~/sglang-env/bin/python scripts/quant_tax/gen.py \
    --task gsm8k --quant Q8_0 --port 8090
"""

import argparse
import json
import os
import time

from lib.quanttax import build_prompt, chat, max_tokens_for, server_health

DATA = "dataset/quant_tax"
OUT = "results/quant_tax"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "mmlu", "humaneval"])
    ap.add_argument("--quant", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    if not server_health(base):
        raise SystemExit(f"no llama-server on {base}")

    rows = [json.loads(l) for l in open(f"{DATA}/{a.task}.jsonl") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    mt = max_tokens_for(a.task)

    gens, t0 = [], time.time()
    for i, r in enumerate(rows):
        text, meta = chat(base, build_prompt(r), mt, a.seed, a.task)
        gens.append({"id": r["id"], "text": text, "meta": meta})
        if (i + 1) % 25 == 0:
            print(f"  {a.task}/{a.quant}: {i+1}/{len(rows)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.quant}__{a.task}.gens.json"
    with open(path, "w") as f:
        json.dump(
            {
                "task": a.task,
                "quant": a.quant,
                "seed": a.seed,
                "max_tokens": mt,
                "n": len(rows),
                "wall_s": round(time.time() - t0, 1),
                "gens": gens,
            },
            f,
        )
    print(f"{a.quant}/{a.task}: {len(rows)} gens in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
