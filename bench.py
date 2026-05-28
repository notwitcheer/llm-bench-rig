#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from lib.config import load_config, get
from lib.meta import extract_metadata, save_metadata
from lib.gpu import get_vram_used_mib
from lib.speed_llama import run_llama_bench
from lib.speed_vllm import run_vllm_bench
from lib.quality import run_quality_bench
from lib.progress import Progress

def run_benchmark(model_path: str, speed_only: bool = False, quality_only: bool = False):
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    model_path = str(Path(model_path).expanduser())

    meta = extract_metadata(Path(model_path))
    out_dir = save_metadata(meta, results_dir)
    slug = meta["slug"]
    engine = meta["engine"]

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {meta['name']}")
    print(f"  Engine: {engine} | Quant: {meta['quant']} | Size: {meta['size_gib']} GiB")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    progress = Progress(out_dir / "progress.json", model=slug)

    try:
        if not quality_only:
            _run_speed(model_path, engine, meta, out_dir, progress)

        if not speed_only:
            _run_quality(model_path, engine, meta, out_dir, progress)

        progress.done()
        print(f"\nBenchmark complete: {slug}")
        print(f"Results: {out_dir}")

    except Exception as e:
        progress.fail(str(e))
        print(f"\nBenchmark FAILED: {e}", file=sys.stderr)
        raise

def _run_speed(model_path, engine, meta, out_dir, progress):
    print("[speed] Measuring VRAM before load...")
    vram_before = get_vram_used_mib()
    progress.update("speed_init", 5)

    if engine == "llama.cpp":
        print("[speed] Running llama-bench...")
        progress.update("speed_llama_bench", 10)
        speed_results = run_llama_bench(model_path)
    else:
        print("[speed] Running vLLM benchmarks...")
        progress.update("speed_vllm_bench", 10)
        speed_results = run_vllm_bench()

    print("[speed] Measuring VRAM after inference...")
    vram_after = get_vram_used_mib()

    speed_results["vram_before_mib"] = vram_before
    speed_results["vram_idle_mib"] = vram_after
    speed_results["vram_peak_mib"] = vram_after
    speed_results["engine"] = engine

    speed_file = out_dir / "speed.json"
    with open(speed_file, "w") as f:
        json.dump(speed_results, f, indent=2)
    print(f"[speed] Results written to {speed_file}")
    progress.update("speed_done", 40, partial=speed_results)

def _run_quality(model_path, engine, meta, out_dir, progress):
    print("[quality] Starting eval harness...")
    progress.update("quality_init", 45)

    quality_results = run_quality_bench(model_path, engine)

    quality_file = out_dir / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(quality_results, f, indent=2)
    print(f"[quality] Results written to {quality_file}")
    progress.update("quality_done", 95, partial=quality_results)

def main():
    parser = argparse.ArgumentParser(description="Benchmark a model on RTX 5090")
    parser.add_argument("model_path", help="Path to .gguf file or safetensors directory")
    parser.add_argument("--speed-only", action="store_true", help="Skip quality benchmarks")
    parser.add_argument("--quality-only", action="store_true", help="Skip speed benchmarks")
    args = parser.parse_args()
    run_benchmark(args.model_path, args.speed_only, args.quality_only)

if __name__ == "__main__":
    main()
