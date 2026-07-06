#!/usr/bin/env python3
"""Aggregate Z-Image-Turbo synth records into a results file (Mac-side).

Reads synth.json (from scripts/zimage_synth.py on capsule), splits by ``kind``
so the shared (1024px, 8-step) point is never double-counted, aggregates each
half via the tested ``lib.zimage.score`` helpers, and writes
results/zimage/z-image-turbo.json for the chart + report.

think:"n/a" — image generation has no think/no-think mode (ADR-0002's gate is
scoped to bench.py::run_benchmark; recorded here only for documentation).

Usage:
  python3 scripts/zimage_bench.py \
    --synth results/zimage/synth.json --out results/zimage/z-image-turbo.json
"""
import argparse
import json

from lib.zimage.score import aggregate

# Config actually run (verified from capsule logs): pure diffusers ZImagePipeline,
# bf16, SDPA attention (no flash-attn), turbo distilled schedule, guidance off,
# batch 1, seed 42, no torch.compile, no quantization.
RUN_CONFIG = {
    "precision": "bf16", "attention": "sdpa (no flash-attn)", "guidance_scale": 0.0,
    "steps_main": 8, "schedule": "turbo distilled (9 call -> 8 DiT forwards)",
    "batch_size": 1, "seed": 42, "torch_compile": False, "quantization": "none",
    "diffusers": "0.37.1", "torch": "2.10.0+cu128", "gpu": "RTX 5090 32GB (sm_120)",
}

# Vendor / reference facts for Z-Image-Turbo (from the model card + project page).
VENDOR_CLAIM = {
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "arch": "6B single-stream DiT + Qwen3-4B text encoder + VAE",
    "steps": "num_inference_steps=9 -> 8 DiT forwards",
    "license": "Apache-2.0",
    "note": "first sm_120 (RTX 5090) numbers; out-of-box bf16, SDPA, no compile or "
            "quantization (a floor — the vendor's own speed uses distilled few-step, "
            "which we run, but not torch.compile).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_image = []
    for path in args.synth:
        per_image.extend(json.load(open(path)))

    ok = [r for r in per_image if r.get("ok")]
    matrix = [r for r in ok if r.get("kind") == "matrix"]
    sweep = [r for r in ok if r.get("kind") == "step_sweep"]
    walls = [r for r in per_image if r.get("kind") == "matrix" and not r.get("ok")]

    agg_m = aggregate(matrix)   # keyed (resolution, 8)
    agg_s = aggregate(sweep)    # keyed (1024, steps)
    by_res = {str(res): cell for (res, st), cell in sorted(agg_m.items())}
    by_steps = {str(st): cell for (res, st), cell in sorted(agg_s.items())}
    vram_walls = sorted({r["resolution"] for r in walls})

    out = {
        "model": "z-image-turbo",
        "think": "n/a",
        "run_config": RUN_CONFIG,
        "vendor_claim": VENDOR_CLAIM,
        "by_resolution": by_res,
        "by_steps": by_steps,
        "vram_wall_resolutions": vram_walls,
        "per_image": per_image,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("--- by resolution (steps=8) ---")
    for res in sorted(by_res, key=int):
        c = by_res[res]
        print(f"  {res:>4}px  n={c['n']}  compute {c['gen_seconds_mean']:.3f}s  "
              f"{c['images_per_min']:6.1f} img/min  peakVRAM {c['peak_vram_mib_max']}MiB")
    if vram_walls:
        print(f"  VRAM wall (OOM) at: {vram_walls}")
    print("--- step sweep @1024 ---")
    for st in sorted(by_steps, key=int):
        c = by_steps[st]
        print(f"  {st:>3} steps  compute {c['gen_seconds_mean']:.3f}s  {c['images_per_min']:.1f} img/min")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
