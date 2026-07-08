#!/usr/bin/env python3
"""Aggregate ltx_synth output into the bench summary (pure CPU, runs anywhere).

Thin CLI over ``lib.ltx.score.summarize`` (tested): per-config mean generation
seconds, seconds-of-compute per second-of-video, and true peak VRAM, plus the
one-off pipeline load time.

Usage:
  python3 scripts/ltx_bench.py --synth-json synth.json --out summary.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ltx.score import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.load(open(args.synth_json))
    summary = summarize(payload["records"])
    summary["load_seconds"] = payload.get("load_seconds")

    json.dump(summary, open(args.out, "w"), indent=2)
    for cfg, s in summary["configs"].items():
        if s["mean_gen_seconds"] is None:
            print(f"{cfg}: {s['n_failed']} failed (OOM?)")
            continue
        print(
            f"{cfg}: n={s['n_ok']}  mean {s['mean_gen_seconds']:.1f}s/clip  "
            f"{s['seconds_per_video_second']:.2f}s per video-second  "
            f"peak {s['max_peak_vram_mib']}MiB"
        )
    print(f"load once: {summary['load_seconds']}s -> wrote {args.out}")


if __name__ == "__main__":
    main()
