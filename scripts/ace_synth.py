#!/usr/bin/env python3
"""ACE-Step 1.5 music-gen benchmark synthesis adapter.

Runs on capsule in ~/ace-step (the ACE-Step clone, its own uv venv) — NOT the
benchmark-rig venv; it imports the `acestep` package. Kept in the rig repo for
reproducibility. DiT-only (no LM), matches the paper's turbo config
(8 steps, shift=3.0, guidance off). Loads one DiT tier once, discards a warm-up
generation, then times each (prompt x duration) with perf_counter (wall, incl.
file save) plus a peak-VRAM sampler, and records the pipeline's own
time_costs breakdown. Emits synth.json for the Mac-side aggregator.

Usage (on capsule, from ~/ace-step):
  ACESTEP_INIT_LLM=false .venv/bin/python ace_synth.py \
    --model-tier 2b --config-path acestep-v15-turbo \
    --prompts prompts.json --durations 30,120,240 --steps 8 --seed 0 \
    --out-dir ~/ace-out --synth-json ~/ace-out/synth-2b.json
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
    def __init__(self, hz: float = 4.0):
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
    ap.add_argument("--model-tier", required=True, help="label for records: 2b / xl")
    ap.add_argument("--config-path", required=True, help="acestep-v15-turbo / acestep-v15-xl-turbo")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--durations", default="30,120,240")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synth-json", required=True)
    ap.add_argument("--save-audio", action="store_true", help="persist wavs (for sample songs)")
    args = ap.parse_args()

    os.environ.setdefault("ACESTEP_INIT_LLM", "false")
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    prompts = json.load(open(args.prompts))
    if args.limit:
        prompts = prompts[:args.limit]
    durations = [float(d) for d in args.durations.split(",")]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    dit = AceStepHandler()
    try:
        dit.initialize_service(project_root=os.getcwd(), config_path=args.config_path,
                               device="cuda", use_flash_attention=False)
    except TypeError:
        # older/newer signature without the kwarg — SDPA is auto-selected anyway
        dit.initialize_service(project_root=os.getcwd(), config_path=args.config_path, device="cuda")
    llm = LLMHandler()  # constructed but NOT initialized — DiT-only

    def gen(caption, lyrics, duration, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = GenerationParams(
            caption=caption, lyrics=lyrics, instrumental=(not lyrics.strip()),
            duration=duration, inference_steps=args.steps, seed=args.seed, shift=3.0,
        )
        c = GenerationConfig(batch_size=1, seeds=[args.seed], use_random_seed=False,
                             audio_format="wav")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = generate_music(dit, llm, p, c, save_dir=save_dir)
        torch.cuda.synchronize()
        return r, time.perf_counter() - t0

    # warm-up (discarded — settles CUDA graphs / lazy init)
    print("warm-up (discarded)...", flush=True)
    gen(prompts[0]["caption"], prompts[0].get("lyrics", ""), 30.0,
        os.path.join(args.out_dir, "_warmup"))

    records = []
    for pr in prompts:
        for d in durations:
            sd = os.path.join(args.out_dir, args.model_tier if args.save_audio else "_scratch")
            with VramSampler() as vs:
                r, wall = gen(pr["caption"], pr.get("lyrics", ""), d, sd)
            ok = bool(getattr(r, "success", False))
            tc = r.extra_outputs.get("time_costs", {}) if ok else {}
            # gen_seconds = pipeline compute time (diffusion+VAE, excl. file save) —
            # matches the paper's methodology for the A100/3090 comparison. wall
            # (perf_counter, incl. save) kept as the honest end-user number.
            compute_s = tc.get("pipeline_total_time", wall)
            wav = r.audios[0]["path"] if ok and r.audios else None
            sr = r.audios[0].get("sample_rate") if wav else None
            rec = {
                "name": pr["name"], "model_tier": args.model_tier,
                "caption": pr["caption"], "lyrics": pr.get("lyrics", ""),
                "duration_s": d, "song_seconds": d, "steps": args.steps, "seed": args.seed,
                "gen_seconds": round(compute_s, 4), "wall_seconds": round(wall, 4),
                "time_costs": tc, "peak_vram_mib": vs.peak, "sr": sr, "wav": wav, "ok": ok,
            }
            records.append(rec)
            print(f"[{args.model_tier}] {pr['name']:>10} {int(d):>3}s -> "
                  f"{wall:6.3f}s wall  peak {vs.peak}MiB  ok={ok}", flush=True)

    Path(args.synth_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.synth_json, "w"), indent=2)
    print(f"wrote {len(records)} records -> {args.synth_json}")


if __name__ == "__main__":
    main()
