#!/usr/bin/env python3
import argparse
import json
import sys
import threading
from pathlib import Path

from lib.config import load_config, get
from lib.meta import extract_metadata, save_metadata
from lib.offload import resolve_offload
from lib.gpu import get_vram_used_mib
from lib.speed_llama import run_llama_bench
from lib.speed_vllm import run_vllm_bench
from lib.quality import run_quality_bench
from lib.provenance import record_provenance
from lib.progress import Progress


class _VramSampler(threading.Thread):
    """Background poller that records peak VRAM while a benchmark runs.

    llama-bench loads and UNLOADS the model around each sub-test, so a single
    before/after reading catches an empty GPU (~2 MiB). We must sample *during*
    the run to see real usage and capture the true peak (weights + KV at the
    longest context tested)."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()  # not _stop: Thread._stop() is called by join()
        self.peak_mib = 0

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.peak_mib = max(self.peak_mib, get_vram_used_mib())
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join(timeout=3)
        return self.peak_mib

def run_benchmark(model_path: str, speed_only: bool = False, quality_only: bool = False):
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    model_path = str(Path(model_path).expanduser())

    meta = extract_metadata(Path(model_path))
    # Record the quality think mode so the result is self-describing (the quality
    # leaderboard splits ON vs OFF — comparing reasoning to non-reasoning is invalid).
    meta["think"] = bool(get("quality.think", True))
    out_dir = save_metadata(meta, results_dir)
    slug = meta["slug"]
    engine = meta["engine"]

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {meta['name']}")
    print(f"  Engine: {engine} | Quant: {meta['quant']} | Size: {meta['size_gib']} GiB")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    offload = resolve_offload(slug, load_config())
    if offload:
        print(f"  Offload override: {offload}")

    progress = Progress(out_dir / "progress.json", model=slug)

    try:
        if not quality_only:
            _run_speed(model_path, engine, meta, out_dir, progress, offload)

        if not speed_only:
            _run_quality(model_path, engine, meta, out_dir, progress, offload)
        else:
            # No server is up on a speed-only run; still record what can be
            # collected offline (gguf sha, harness sha, versions). Never raises.
            record_provenance(out_dir, None, model_path, None, None)

        progress.done()
        print(f"\nBenchmark complete: {slug}")
        print(f"Results: {out_dir}")

    except Exception as e:
        progress.fail(str(e))
        print(f"\nBenchmark FAILED: {e}", file=sys.stderr)
        raise

def _run_speed(model_path, engine, meta, out_dir, progress, offload=None):
    offload = offload or {}
    print("[speed] Measuring VRAM before load...")
    vram_before = get_vram_used_mib()
    progress.update("speed_init", 5)

    if engine == "llama.cpp":
        print("[speed] Running llama-bench...")
        progress.update("speed_llama_bench", 10)
        sampler = _VramSampler()
        sampler.start()
        try:
            speed_results = run_llama_bench(model_path, **offload)
        finally:
            vram_peak = sampler.stop()  # peak captured DURING the run
    else:
        # offload is llama.cpp-specific; vLLM manages layers internally
        print("[speed] Running vLLM benchmarks...")
        progress.update("speed_vllm_bench", 10)
        speed_results = run_vllm_bench()
        vram_peak = get_vram_used_mib()  # vLLM server is still resident here

    print("[speed] Measuring VRAM after inference...")
    vram_after = get_vram_used_mib()

    # vram_before = empty baseline; vram_peak = real peak sampled mid-run.
    # llama-bench's load/unload cycling prevents a clean steady-state "idle"
    # read, so report peak as the headline VRAM figure.
    speed_results["vram_before_mib"] = vram_before
    speed_results["vram_peak_mib"] = max(vram_peak, vram_after)
    speed_results["vram_idle_mib"] = speed_results["vram_peak_mib"]
    speed_results["engine"] = engine

    speed_file = out_dir / "speed.json"
    with open(speed_file, "w") as f:
        json.dump(speed_results, f, indent=2)
    print(f"[speed] Results written to {speed_file}")
    progress.update("speed_done", 40, partial=speed_results)

def _run_quality(model_path, engine, meta, out_dir, progress, offload=None):
    offload = offload or {}
    print("[quality] Starting eval harness...")
    progress.update("quality_init", 45)

    quality_results = run_quality_bench(model_path, engine, results_dir=out_dir, offload=offload)

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
