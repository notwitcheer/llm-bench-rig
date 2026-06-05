#!/usr/bin/env python3
"""Concurrent aggregate throughput for an OpenAI-compatible endpoint (vLLM or llama-server).

Fires N simultaneous fixed-length chat completions and reports aggregate tok/s = total output
tokens / wall time, at several concurrency levels. Same client for both engines = fair.

Usage: bench_concurrent.py --base http://127.0.0.1:8000 --model <id> --out out.json [--levels 1,8,16,32]
"""
import argparse
import asyncio
import json
import time

import httpx

PROMPT = "Write a detailed paragraph about the history of computing, covering key milestones."


def aggregate_tps(total_output_tokens, wall_s):
    return round(total_output_tokens / wall_s, 2) if wall_s > 0 else 0.0


async def _one(client, base, model, max_tokens):
    r = await client.post(f"{base.rstrip('/')}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
    })
    r.raise_for_status()
    return r.json().get("usage", {}).get("completion_tokens", 0)


async def _run_level(base, model, concurrency, max_tokens):
    async with httpx.AsyncClient(timeout=600) as client:
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[_one(client, base, model, max_tokens) for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    total = sum(outs)
    return {
        "concurrency": concurrency,
        "requests": len(outs),
        "total_output_tokens": total,
        "wall_s": round(wall, 3),
        "aggregate_tps": aggregate_tps(total, wall),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [asyncio.run(_run_level(a.base, a.model, int(c), a.max_tokens)) for c in a.levels.split(",")]
    json.dump(res, open(a.out, "w"), indent=2)
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
