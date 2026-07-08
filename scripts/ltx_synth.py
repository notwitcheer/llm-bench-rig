#!/usr/bin/env python3
"""LTX-2.3 audio-video benchmark adapter (runs on capsule, inside ~/LTX-2/.venv).

Constructs the vendor ``DistilledPipeline`` ONCE (fp8-cast, offload none, the
gate-passing config), then loops prompts x resolution configs, encoding each
clip to mp4. ``gen_seconds`` spans pipeline call THROUGH ``encode_video``: the
video decoder returns a lazy chunk iterator, so VAE decode work only happens
inside the encode — timing the pipeline call alone would undercount. Per-clip
VRAM is torch ``max_memory_allocated`` after ``reset_peak_memory_stats`` (the
caching-allocator-immune counter), with a background nvidia-smi sampler kept as
global context. Model-load time is recorded once, separately. A CUDA OOM at a
config is recorded as a finding, not a crash.

Mirrors ``ltx_pipelines/distilled.py``'s main() exactly for correctness;
mirrors ``scripts/zimage_synth.py`` for measurement conventions.

Usage (on capsule):
  cd ~/LTX-2 && ./.venv/bin/python ~/benchmark-rig/scripts/ltx_synth.py \
    --distilled-checkpoint ~/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root ~/models/gemma-3-12b-qat \
    --upsampler ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --prompts ~/benchmark-rig/dataset/ltx/prompts.json \
    --configs 768x512x97,1280x704x97 --fps 24 --seed 42 \
    --warmup 512x320x33 --out-dir ~/ltx-out/bench \
    --synth-json ~/ltx-out/bench/synth.json
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ltx.config import validate_video_config  # noqa: E402
from lib.ltx.dataset import load_prompts  # noqa: E402


def gpu_vram_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


class VramSampler:
    """Background nvidia-smi peak sampler — global context only (see torch counter)."""
    def __init__(self, hz: float = 2.0):
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
        self._t.join(timeout=2)


def parse_config(s: str):
    w, h, f = (int(x) for x in s.split("x"))
    validate_video_config(width=w, height=h, num_frames=f)
    return w, h, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled-checkpoint", required=True)
    ap.add_argument("--gemma-root", required=True)
    ap.add_argument("--upsampler", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--configs", default="768x512x97,1280x704x97",
                    help="comma-separated WxHxFRAMES entries")
    ap.add_argument("--warmup", default="512x320x33",
                    help="small discarded gen to settle lazy init/autotune")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synth-json", required=True)
    args = ap.parse_args()

    configs = [parse_config(s) for s in args.configs.split(",")]
    warm_w, warm_h, warm_f = parse_config(args.warmup)
    prompts = load_prompts(args.prompts, limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    print("constructing DistilledPipeline once (fp8-cast, offload none)...", flush=True)
    t0 = time.perf_counter()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.upsampler,
        loras=(),
        quantization=QuantizationKind("fp8-cast").to_policy(
            checkpoint_path=args.distilled_checkpoint
        ),
        offload_mode=OffloadMode.NONE,
    )
    load_seconds = round(time.perf_counter() - t0, 2)
    print(f"pipeline constructed in {load_seconds}s", flush=True)
    tiling_config = TilingConfig.default()

    @torch.inference_mode()
    def gen(prompt: str, w: int, h: int, frames: int, out_path: str):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video, audio = pipeline(
            prompt=prompt, seed=args.seed, height=h, width=w,
            num_frames=frames, frame_rate=args.fps, images=[],
            tiling_config=tiling_config,
        )
        encode_video(
            video=video, fps=args.fps, audio=audio, output_path=out_path,
            video_chunks_number=get_video_chunks_number(frames, tiling_config),
        )
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        peak_mib = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
        return gen_s, peak_mib

    print(f"warm-up {warm_w}x{warm_h}x{warm_f} (discarded)...", flush=True)
    gen(prompts[0]["prompt"], warm_w, warm_h, warm_f, str(out_dir / "warmup.mp4"))

    records = []
    for w, h, frames in configs:
        for pr in prompts:
            clip = out_dir / f"{pr['name']}-{w}x{h}-{frames}f.mp4"
            base = {
                "name": pr["name"], "category": pr["category"],
                "width": w, "height": h, "num_frames": frames,
                "fps": args.fps, "seed": args.seed,
            }
            try:
                with VramSampler() as vs:
                    gen_s, peak = gen(pr["prompt"], w, h, frames, str(clip))
                records.append({**base, "gen_seconds": round(gen_s, 3),
                                "peak_vram_mib": peak, "nvidia_global_mib": vs.peak,
                                "clip": str(clip), "ok": True})
                print(f"[{w}x{h}] {pr['name']:>14} -> {gen_s:7.2f}s  "
                      f"peak {peak}MiB (nvidia {vs.peak}MiB)", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                records.append({**base, "gen_seconds": None, "peak_vram_mib": None,
                                "nvidia_global_mib": None, "clip": None, "ok": False,
                                "error": "cuda_oom"})
                print(f"[{w}x{h}] {pr['name']:>14} -> CUDA OOM (VRAM wall)", flush=True)

    payload = {"load_seconds": load_seconds, "records": records}
    Path(args.synth_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.synth_json, "w"), indent=2)
    print(f"wrote {len(records)} records -> {args.synth_json}", flush=True)


if __name__ == "__main__":
    main()
