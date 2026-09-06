#!/usr/bin/env python3
"""Served (http) speed lane: time streaming chat completions against a live server.

Usage:
    python3 scripts/speed_served.py --api http://127.0.0.1:8090 --mode base \
        --out results/served/base.jsonl [--repeats 1] [--max-tokens 256] \
        [--no-cache-prompt] [--summary-only]

Sends one warm-up request (discarded), then the four fixed workloads from
lib/workloads.py (prose, code, repetitive, chat; 8 prompts each), `--repeats`
passes, at temperature 0. Every request is appended to --out as one json line
stamped with mode, cache_prompt, workload, idx and rep. At the end the file is
re-read and a p50/p90 summary (ttft, perceived tps, total tps, server predicted
tps, acceptance rate when speculation is on) is printed per workload and overall.

--summary-only skips the run and summarises an existing --out file, so numbers
in a report always come from the raw records, never from an in-run aggregate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.speed_served import load_jsonl, run_served_lane, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True, help="server root or root/v1, e.g. http://127.0.0.1:8090")
    ap.add_argument("--mode", required=True, help="server flag set label stamped on every record, e.g. base|mtp_n2|ngram")
    ap.add_argument("--out", required=True, help="jsonl file, appended to")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-cache-prompt", action="store_true", help="send cache_prompt=false (cold prefill every request)")
    ap.add_argument("--summary-only", action="store_true", help="do not run; summarise the existing --out file")
    a = ap.parse_args()

    out = Path(a.out)
    if not a.summary_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        run_served_lane(a.api, a.mode, out, repeats=a.repeats, max_tokens=a.max_tokens,
                        cache_prompt=not a.no_cache_prompt)
        print(f"LANE_DONE served {a.mode} -> {out}", flush=True)

    records = [r for r in load_jsonl(out) if r.get("mode") == a.mode]
    if not records:
        print(f"no records for mode {a.mode!r} in {out}", file=sys.stderr)
        return 1
    report = {"mode": a.mode, "overall": summarise(records),
              "by_workload": summarise(records, group_by="workload")}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
