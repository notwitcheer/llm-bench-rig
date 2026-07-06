#!/usr/bin/env python3
"""Z-Image-Turbo few-step text-to-image benchmark adapter (runs on capsule).

Loads ``ZImagePipeline`` once (bf16, SDPA — no flash-attn), discards a warm-up
generation, then times each (prompt x resolution) at a fixed step count with a
cuda-synchronized ``perf_counter`` (``gen_seconds`` = pure pipeline compute)
plus ``wall_seconds`` (incl. PNG save) and a background peak-VRAM sampler. A
second pass sweeps the step count at one resolution to expose the few-step
economy (each step = one DiT forward). Emits synth.json for the Mac aggregator.

Kept in the rig repo for reproducibility; imports ``diffusers`` from the
capsule's torch-2.10+cu128 env (no repo of its own — Z-Image is a pure diffusers
pipeline). Records carry ``kind`` = "matrix" | "step_sweep" so the aggregator
never double-counts the shared (1024px, 8-step) point.

Usage (on capsule):
  ~/unsloth-env/bin/python zimage_synth.py \
    --model ~/models/z-image-turbo --prompts prompts.json \
    --resolutions 512,1024,1536,2048 --steps 8 --step-sweep 4,8,9,16 \
    --sweep-resolution 1024 --seed 42 --out-dir ~/zimage-out \
    --synth-json ~/zimage-out/synth.json --save-images
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def gpu_vram_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


class VramSampler:
    """Background peak-VRAM sampler — a before/after read misses the peak."""
    def __init__(self, hz: float = 5.0):
        self.peak = gpu_vram_used_mib()
        self._stop = False
        self._interval = 1.0 / hz
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, gpu_vram_used_mib())
            time.sleep(self._interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--resolutions", default="512,1024,1536,2048")
    ap.add_argument("--steps", type=int, default=8, help="distilled turbo default (9 call -> 8 DiT forwards)")
    ap.add_argument("--step-sweep", default="4,8,9,16")
    ap.add_argument("--sweep-resolution", type=int, default=1024)
    ap.add_argument("--sweep-prompt-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synth-json", required=True)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--save-resolution", type=int, default=1024,
                    help="persist the showcase PNG grid only at this resolution")
    args = ap.parse_args()

    import torch
    from diffusers import ZImagePipeline

    prompts = json.load(open(args.prompts))
    if args.limit:
        prompts = prompts[:args.limit]
    resolutions = [int(x) for x in args.resolutions.split(",")]
    img_dir = os.path.join(args.out_dir, "samples")
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    print(f"loading ZImagePipeline from {args.model} ...", flush=True)
    pipe = ZImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    def gen(prompt, res, steps, save_path=None):
        # reset torch's peak tracker so max_memory_allocated() is THIS gen's true
        # tensor peak (model-resident + activations), immune to the caching
        # allocator's retained high-water mark that pollutes nvidia-smi reads.
        g = torch.Generator("cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        image = pipe(
            prompt=prompt, height=res, width=res, num_inference_steps=steps,
            guidance_scale=0.0, generator=g,
        ).images[0]
        torch.cuda.synchronize()
        compute_s = time.perf_counter() - t0
        torch_peak_mib = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if save_path:
            image.save(save_path)
        wall_s = time.perf_counter() - t0
        return compute_s, wall_s, torch_peak_mib

    # warm-up (discarded — settles CUDA graphs / lazy init / kernel autotune)
    print("warm-up (discarded)...", flush=True)
    gen(prompts[0]["prompt"], 1024, args.steps)

    records = []

    def record(pr, res, steps, compute_s, wall_s, peak, nvidia_global, kind, saved):
        rec = {
            "name": pr["name"], "category": pr.get("category", ""),
            "resolution": res, "width": res, "height": res, "steps": steps,
            "seed": args.seed, "gen_seconds": round(compute_s, 4),
            "wall_seconds": round(wall_s, 4),
            "peak_vram_mib": peak,            # torch max_memory_allocated (true per-gen)
            "nvidia_global_mib": nvidia_global,  # nvidia-smi total incl allocator cache
            "kind": kind, "image": saved, "ok": True,
        }
        records.append(rec)
        print(f"[{kind}] {pr['name']:>12} {res:>4}px {steps:>2}st -> "
              f"{compute_s:6.3f}s compute  {wall_s:6.3f}s wall  "
              f"peak {peak}MiB (nvidia {nvidia_global}MiB)", flush=True)

    # ---- main matrix: prompts x resolutions @ fixed distilled steps ----
    for pr in prompts:
        for res in resolutions:
            saved = None
            if args.save_images and res == args.save_resolution:
                saved = os.path.join(img_dir, f"{pr['name']}-{res}-{args.steps}st.png")
            try:
                with VramSampler() as vs:
                    compute_s, wall_s, tpeak = gen(pr["prompt"], res, args.steps, saved)
                record(pr, res, args.steps, compute_s, wall_s, tpeak, vs.peak, "matrix", saved)
            except torch.cuda.OutOfMemoryError:
                # a VRAM wall at high resolution is itself a finding — record it
                torch.cuda.empty_cache()
                records.append({
                    "name": pr["name"], "category": pr.get("category", ""),
                    "resolution": res, "width": res, "height": res, "steps": args.steps,
                    "seed": args.seed, "gen_seconds": None, "wall_seconds": None,
                    "peak_vram_mib": None, "kind": "matrix", "image": None, "ok": False,
                    "error": "cuda_oom",
                })
                print(f"[matrix] {pr['name']:>12} {res:>4}px -> CUDA OOM (VRAM wall)", flush=True)

    # ---- step sweep: one prompt x {steps} @ one resolution (few-step economy) ----
    sweep_steps = [int(x) for x in args.step_sweep.split(",")]
    spr = prompts[args.sweep_prompt_index]
    for st in sweep_steps:
        saved = None
        if args.save_images:
            saved = os.path.join(img_dir, f"sweep-{spr['name']}-{args.sweep_resolution}-{st}st.png")
        with VramSampler() as vs:
            compute_s, wall_s, tpeak = gen(spr["prompt"], args.sweep_resolution, st, saved)
        record(spr, args.sweep_resolution, st, compute_s, wall_s, tpeak, vs.peak, "step_sweep", saved)

    Path(args.synth_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.synth_json, "w"), indent=2)
    print(f"wrote {len(records)} records -> {args.synth_json}")


if __name__ == "__main__":
    main()
