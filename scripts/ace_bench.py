#!/usr/bin/env python3
"""Aggregate ACE-Step music-gen synth records into a results file (Mac-side).

Reads one or more synth-<tier>.json (from scripts/ace_synth.py on capsule),
aggregates per (model_tier, duration_s) via the tested lib.ace_step.score
helpers, and writes results/ace_step/ace-step-music.json for the chart + report.

think:"n/a" — music generation has no think/no-think mode (ADR-0002's gate is
scoped to bench.py::run_benchmark; recorded here only for documentation).

Usage:
  python3 scripts/ace_bench.py \
    --synth results/ace_step/synth-2b.json results/ace_step/synth-xl.json \
    --out results/ace_step/ace-step-music.json
"""
import argparse
import json

from lib.ace_step.score import aggregate

# Config actually run (verified from capsule logs): DiT-only, bf16, SDPA attention,
# turbo forces guidance 1.0 (no CFG), 8 steps, shift 3.0, batch 1, seed 0, no
# torch.compile (warm-up ~1.1s), no quantization (~9/15 GB = bf16 footprint).
RUN_CONFIG = {
    "dit_only": True, "precision": "bf16", "attention": "sdpa",
    "inference_steps": 8, "shift": 3.0, "guidance": "off (turbo, CFG=1.0)",
    "batch_size": 1, "seed": 0, "torch_compile": False, "quantization": "none",
    "sample_rate_hz": 48000, "channels": "stereo",
}

# Vendor claims for ACE-Step 1.5 (2B turbo), from the project page + paper
# (arXiv 2602.00744). "a song" = a 240 s track, 8-step turbo DiT, bf16.
VENDOR_CLAIM = {
    "model": "acestep-v15-turbo (2B)",
    "song_seconds": 240,
    "a100_seconds": "~1-2 (paper ~1s, project page <2s)",
    "a100_rtf": "~120-240x",
    "rtx3090_seconds": "<10",
    "rtx3090_rtf": "~24x",
    "note": "vendor numbers are on their hardware/config; the 5090 figures here "
            "are out-of-box bf16 with no torch.compile or quantization (a floor).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_song = []
    for path in args.synth:
        per_song.extend(json.load(open(path)))

    agg = aggregate(per_song)  # keyed by (model_tier, duration_s)

    tiers: dict = {}
    for (tier, dur), cell in agg.items():
        tiers.setdefault(tier, {})[str(int(dur))] = cell

    out = {
        "model": "ace-step-1.5",
        "think": "n/a",
        "run_config": RUN_CONFIG,
        "vendor_claim": VENDOR_CLAIM,
        "tiers": tiers,
        "per_song": per_song,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # human-readable summary
    for tier in sorted(tiers):
        print(f"--- {tier} ---")
        for d in sorted(tiers[tier], key=int):
            c = tiers[tier][d]
            print(f"  {d:>3}s  n={c['n']}  compute {c['gen_seconds_mean']:.3f}s  "
                  f"RTF {c['rtf_mean']:.1f}x  peakVRAM {c['peak_vram_mib_max']}MiB")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
