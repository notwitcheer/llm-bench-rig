#!/usr/bin/env python3
"""Recipe bench: the served lane for one server flag set, agent-shaped.

Runs against a live llama-server (or any OpenAI-compatible /v1 with `cache_prompt`
honoured) and appends one json line per request to --out, stamped with --mode:

  1. the four short workloads from lib/workloads.py (prose, code, repetitive, chat;
     8 prompts each, 256 tokens, temperature 0) via lib.speed_served.run_served_lane;
  2. the LONG lane: a system prompt of exactly --sys-tokens tokens (a synthetic
     repository dump, sized with the server's /tokenize so it is the same token count
     on every model) followed by 8 short coding questions, sent twice:
       - cold: cache_prompt=false, the whole 16k prompt is prefilled every request
         (the first turn of an agent session, or a server without prompt caching);
       - warm: cache_prompt=true after one primer request with the same system prompt,
         so only the question is prefilled (every later turn of the session).

Workload labels: prose|code|repetitive|chat|long_cold|long_warm. The summary printed
at the end (and written to --summary json) carries per-workload p50/p90 TTFT,
perceived TPS and total TPS, plus acceptance when the server reports draft counts.
Vocabulary and percentiles follow the rig's benchmarking standard (TTFT = first content
chunk as the client sees it; perceived TPS = completion_tokens / (total_s - ttft_s)).

Usage:
  python3 scripts/recipe_bench.py --api http://127.0.0.1:8093 --mode fa_q8kv \
      --out results/<slug>/recipe/fa_q8kv.jsonl --sys-tokens 16384 [--summary path.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.speed_served import (append_jsonl, load_jsonl, run_served_lane,  # noqa: E402
                              stream_chat, summarise)

LONG_QUESTIONS = [
    "which module defines the function that validates a checksum, and what does it return on failure?",
    "list the three modules whose retry limit is above 5, with the limit for each.",
    "write a one-paragraph summary of what this repository does.",
    "which function has the largest timeout constant, and what is the value?",
    "is there any module that logs at debug level inside its main loop? name it.",
    "propose a name for a new module that would sit between the parser and the exporter, and one sentence on why.",
    "how many modules are in the dump? answer with a number and how you counted.",
    "which module would you refactor first, and why, in two sentences?",
]


def repo_dump(n_modules: int) -> str:
    """Deterministic synthetic repository: n small python modules with varying constants."""
    parts = ["# repository context (read-only). answer questions about it.\n"]
    for i in range(n_modules):
        retry = (i * 7) % 11
        timeout = 5 + (i * 13) % 90
        level = "debug" if i % 9 == 4 else "info"
        parts.append(
            f"\n## file src/module_{i:03d}.py\n"
            f"import logging\nlog = logging.getLogger(__name__)\n"
            f"RETRY_LIMIT = {retry}\nTIMEOUT_S = {timeout}\n\n"
            f"def process_{i:03d}(items, checksum):\n"
            f"    \"\"\"process a batch of {retry + 2} items and verify the checksum.\"\"\"\n"
            f"    total = 0\n"
            f"    for n, item in enumerate(items):\n"
            f"        log.{level}('module {i:03d} item %d', n)\n"
            f"        total += (hash(item) ^ {i * 31 + 7}) % 1000\n"
            f"        if n > RETRY_LIMIT:\n"
            f"            break\n"
            f"    if total % 997 != checksum:\n"
            f"        return None\n"
            f"    return total\n"
        )
    return "".join(parts)


def tokenize(api: str, text: str, client: httpx.Client) -> list[int] | None:
    root = api.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        r = client.post(f"{root}/tokenize", json={"content": text}, timeout=120)
        r.raise_for_status()
        return r.json().get("tokens")
    except Exception:
        return None


def detokenize(api: str, tokens: list[int], client: httpx.Client) -> str | None:
    root = api.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        r = client.post(f"{root}/detokenize", json={"tokens": tokens}, timeout=120)
        r.raise_for_status()
        return r.json().get("content")
    except Exception:
        return None


def build_system_prompt(api: str, target_tokens: int, client: httpx.Client, log=print) -> tuple[str, int, str]:
    """Return (text, token_count, method). Grows the dump until it exceeds the target,
    then trims to exactly target_tokens via /tokenize + /detokenize. Falls back to a
    4-chars-per-token estimate if the server has no tokenizer endpoints."""
    n = max(8, target_tokens // 120)
    text = repo_dump(n)
    toks = tokenize(api, text, client)
    if toks is None:
        est = text[: target_tokens * 4]
        log(f"[recipe] /tokenize unavailable, system prompt sized by 4 chars/token: {len(est)} chars")
        return est, len(est) // 4, "estimate"
    while len(toks) < target_tokens:
        n = int(n * 1.3) + 1
        text = repo_dump(n)
        nxt = tokenize(api, text, client)
        if nxt is None:
            return text, len(toks), "tokenize-untrimmed"
        toks = nxt
    trimmed = detokenize(api, toks[:target_tokens], client)
    if trimmed is None:
        return text, len(toks), "tokenize-untrimmed"
    check = tokenize(api, trimmed, client) or []
    log(f"[recipe] system prompt {len(check)} tokens (target {target_tokens}, {n} modules, {len(trimmed)} chars)")
    return trimmed, len(check), "tokenize"


def run_long_lane(api: str, mode: str, out, system: str, sys_tokens: int, max_tokens: int,
                  log=print, chat=stream_chat) -> list[dict]:
    stamp = {"mode": mode, "max_tokens": max_tokens, "sys_tokens": sys_tokens,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    recs = []
    # cold: every request prefills the whole prompt
    for i, q in enumerate(LONG_QUESTIONS):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        r = chat(api, msgs, max_tokens, cache_prompt=False)
        r.update(stamp, workload="long_cold", idx=i, rep=0, cache_prompt=False)
        append_jsonl(out, r); recs.append(r)
        t = r.get("timings") or {}
        log(f"[{mode}] long_cold#{i}: ttft {r['ttft_s']}s total {r['total_s']}s tok {r['completion_tokens']} "
            f"cache_n {t.get('cache_n')} pred_tps {t.get('predicted_per_second')}")
    # warm: one primer, then 8 questions against the cached prefix
    chat(api, [{"role": "system", "content": system}, {"role": "user", "content": "ready?"}], 8, cache_prompt=True)
    for i, q in enumerate(LONG_QUESTIONS):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        r = chat(api, msgs, max_tokens, cache_prompt=True)
        r.update(stamp, workload="long_warm", idx=i, rep=0, cache_prompt=True)
        append_jsonl(out, r); recs.append(r)
        t = r.get("timings") or {}
        log(f"[{mode}] long_warm#{i}: ttft {r['ttft_s']}s total {r['total_s']}s tok {r['completion_tokens']} "
            f"cache_n {t.get('cache_n')} pred_tps {t.get('predicted_per_second')}")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sys-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--summary", default=None, help="write the summary json here as well")
    ap.add_argument("--skip-short", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.summary_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not a.skip_short:
            run_served_lane(a.api, a.mode, out, repeats=1, max_tokens=a.max_tokens, cache_prompt=True)
        if not a.skip_long:
            with httpx.Client(timeout=300) as c:
                system, n_tok, method = build_system_prompt(a.api, a.sys_tokens, c)
            run_long_lane(a.api, a.mode, out, system, n_tok, a.max_tokens)
        print(f"LANE_DONE recipe {a.mode} -> {out}", flush=True)
    records = [r for r in load_jsonl(out) if r.get("mode") == a.mode]
    if not records:
        print(f"no records for mode {a.mode!r} in {out}", file=sys.stderr)
        return 1
    report = {"mode": a.mode, "n_records": len(records), "overall_short": summarise(
        [r for r in records if not str(r.get("workload", "")).startswith("long")]),
        "by_workload": summarise(records, group_by="workload")}
    txt = json.dumps(report, indent=1)
    print(txt)
    if a.summary:
        Path(a.summary).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
