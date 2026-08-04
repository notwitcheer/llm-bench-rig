"""t090: speed and VRAM for one quant, against a running llama-server.

Prefill and decode are reported separately because quantization moves them in
different directions: decode is memory-bandwidth bound and gains from a smaller
model, prefill is compute bound and gains much less. A single tok/s number hides that.

Timings come from llama-server's own block rather than wall clock, so HTTP and
JSON overhead stay out of the measurement. The warm-up run is discarded; the
remaining N are reported with their spread, per the design's validity checklist.

Run (capsule, server up):
  PYTHONPATH=. ~/sglang-env/bin/python scripts/quant_tax/speed.py --quant Q8_0 --port 8090
"""

import argparse
import json
import os
import statistics
import subprocess

from lib.quanttax import complete, server_health

OUT = "results/quant_tax"

# Fixed prompt, identical across the ladder. Long enough that prefill is measurable.
PROMPT = (
    "Write a detailed technical explanation of how a modern GPU executes a matrix "
    "multiplication, covering memory hierarchy, tiling, and occupancy. "
) * 8


def vram_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--runs", type=int, default=3, help="timed runs after the warm-up")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    if not server_health(base):
        raise SystemExit(f"no llama-server on {base}")

    vram = vram_used_mib()

    complete(base, PROMPT, a.n_predict, a.seed)  # warm-up, discarded

    prefill, decode = [], []
    for _ in range(a.runs):
        _, t = complete(base, PROMPT, a.n_predict, a.seed)
        # llama-server reports *_per_second directly; recompute as a cross-check.
        prefill.append(t["prompt_n"] / (t["prompt_ms"] / 1000.0))
        decode.append(t["predicted_n"] / (t["predicted_ms"] / 1000.0))

    def summarise(xs):
        return {
            "mean": round(statistics.mean(xs), 2),
            "min": round(min(xs), 2),
            "max": round(max(xs), 2),
            "spread_pct": round((max(xs) - min(xs)) / statistics.mean(xs) * 100, 3),
            "runs": [round(x, 2) for x in xs],
        }

    rec = {
        "quant": a.quant,
        "runs": a.runs,
        "n_predict": a.n_predict,
        "vram_used_mib": vram,
        "prefill_tok_s": summarise(prefill),
        "decode_tok_s": summarise(decode),
    }

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/speed.json"
    existing = json.load(open(path)) if os.path.exists(path) else []
    keyed = {r["quant"]: r for r in existing}
    keyed[a.quant] = rec
    with open(path, "w") as f:
        json.dump(sorted(keyed.values(), key=lambda r: r["quant"]), f, indent=2)

    print(
        f"{a.quant:8s} prefill {rec['prefill_tok_s']['mean']:8.2f} tok/s "
        f"(±{rec['prefill_tok_s']['spread_pct']:.2f}%)  "
        f"decode {rec['decode_tok_s']['mean']:7.2f} tok/s "
        f"(±{rec['decode_tok_s']['spread_pct']:.2f}%)  VRAM {vram} MiB"
    )
    print(f"speed -> {path}")


if __name__ == "__main__":
    main()
