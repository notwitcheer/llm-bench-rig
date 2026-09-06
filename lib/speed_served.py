"""Served (http) speed lane: time streaming chat completions against a running server.

llama-bench measures the engine in isolation; this lane measures what a client
sees through llama-server's OpenAI-compatible streaming endpoint. Per request it
records:

  ttft_s             wall time from send to the first content chunk
  total_s            wall time from send to the last chunk
  completion_tokens  from the final usage object (stream_options.include_usage)
  prompt_tokens      likewise
  timings            llama-server's own object when present (prompt_n, prompt_ms,
                     predicted_n, predicted_ms, predicted_per_second, draft_n,
                     draft_n_accepted when speculation is on)
  sha256             of the generated text, so two server flag sets can be checked
                     for byte-identical output at temperature 0

Vocabulary used in the summary and in dataset/README.md:

  TTFT           ttft_s as above (prefill plus queueing, as perceived)
  perceived TPS  completion_tokens / (total_s - ttft_s): the rate the reader
                 watches text arrive at, after the first token
  total TPS      completion_tokens / total_s: throughput including the wait

Percentiles (p50, p90) are reported, never means; one warm-up request is sent
and discarded before recording. Every record carries the `mode` label (server
flag set, e.g. base or mtp_n2), `cache_prompt`, workload name, prompt index and
repeat, and is appended as one json line so the summary can always be rebuilt
from the raw records.

The stream parser and the summariser are pure functions and unit tested with
canned chunks; only `stream_chat` touches the network.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Iterable, Iterator

import httpx

from lib.workloads import WORKLOADS

DEFAULT_MAX_TOKENS = 256
WARMUP_PROMPT = "hi"
WARMUP_TOKENS = 8


# --- request body ---

def build_request(messages: list[dict], max_tokens: int, cache_prompt: bool = True,
                  think: bool = False) -> dict:
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": cache_prompt,
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def chat_completions_url(api: str) -> str:
    """Accept either a server root or a root ending in /v1."""
    root = api.rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return root + "/chat/completions"


# --- sse parsing (pure) ---

def parse_sse_stream(lines: Iterable[str], clock: Callable[[], float] = time.time,
                     t0: float | None = None) -> dict:
    """Fold an SSE line stream into one timing record.

    `lines` yields raw text lines as received (with or without trailing newline).
    `clock` is called when the first content chunk arrives and at the end, so a
    fake clock makes the parser deterministic in tests. The record's ttft_s is
    None when no content chunk ever arrived (empty answer, or all reasoning).
    """
    if t0 is None:
        t0 = clock()
    first = None
    text_parts: list[str] = []
    usage = None
    timings = None
    finish_reason = None
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("timings"), dict):
            timings = d["timings"]
        if isinstance(d.get("usage"), dict):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            content = delta.get("content")
            if content:
                if first is None:
                    first = clock()
                text_parts.append(content)
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
    end = clock()
    text = "".join(text_parts)
    return {
        "ttft_s": round(first - t0, 4) if first is not None else None,
        "total_s": round(end - t0, 4),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "timings": timings,
        "finish_reason": finish_reason,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_head": text[:120],
        "has_think_tag": "<think>" in text,
    }


# --- network ---

def stream_chat(api: str, messages: list[dict], max_tokens: int,
                cache_prompt: bool = True, timeout: float = 900,
                client: httpx.Client | None = None) -> dict:
    """POST one streaming chat completion and return its timing record."""
    body = build_request(messages, max_tokens, cache_prompt=cache_prompt)
    url = chat_completions_url(api)
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        t0 = time.time()
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            return parse_sse_stream(resp.iter_lines(), t0=t0)
    finally:
        if own:
            client.close()


def iter_workload_requests(workloads: dict | None = None, repeats: int = 1
                           ) -> Iterator[tuple[int, str, int, str]]:
    """Yield (rep, workload, idx, prompt) in the fixed order the reports use."""
    workloads = WORKLOADS if workloads is None else workloads
    for rep in range(repeats):
        for wl, prompts in workloads.items():
            for i, p in enumerate(prompts):
                yield rep, wl, i, p


def append_jsonl(path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_served_lane(api: str, mode: str, out, repeats: int = 1,
                    max_tokens: int = DEFAULT_MAX_TOKENS, cache_prompt: bool = True,
                    workloads: dict | None = None, log=print,
                    chat: Callable[..., dict] = stream_chat) -> list[dict]:
    """Warm up once (discarded), then record every workload prompt as jsonl.

    `chat` is injectable so the loop is testable without a server.
    """
    stamp = {
        "mode": mode,
        "cache_prompt": cache_prompt,
        "max_tokens": max_tokens,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    chat(api, [{"role": "user", "content": WARMUP_PROMPT}], WARMUP_TOKENS,
         cache_prompt=cache_prompt)  # warm-up: model load, kernel jit; never recorded
    records = []
    for rep, wl, i, prompt in iter_workload_requests(workloads, repeats):
        r = chat(api, [{"role": "user", "content": prompt}], max_tokens,
                 cache_prompt=cache_prompt)
        r.update(stamp, workload=wl, idx=i, rep=rep)
        append_jsonl(out, r)
        records.append(r)
        t = r.get("timings") or {}
        log(f"[{mode}] {wl}#{i} rep{rep}: ttft {r['ttft_s']}s total {r['total_s']}s "
            f"tok {r['completion_tokens']} pred_tps {t.get('predicted_per_second')} "
            f"draft {t.get('draft_n')}/{t.get('draft_n_accepted')}")
    return records


# --- summary (pure) ---

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile on a sorted copy; None for an empty list.

    q in [0, 100]. Linear interpolation between the two nearest ranks, which
    is what numpy's default does, so p50 of [1, 2, 3, 4] is 2.5.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * frac)


def perceived_tps(rec: dict) -> float | None:
    """completion_tokens / (total_s - ttft_s); None when not computable."""
    n = rec.get("completion_tokens")
    total, ttft = rec.get("total_s"), rec.get("ttft_s")
    if not n or total is None or ttft is None:
        return None
    dt = total - ttft
    if dt <= 0:
        return None
    return n / dt


def total_tps(rec: dict) -> float | None:
    n, total = rec.get("completion_tokens"), rec.get("total_s")
    if not n or not total or total <= 0:
        return None
    return n / total


def summarise(records: list[dict], group_by: str | None = None) -> dict:
    """p50/p90 ttft and perceived tps, server predicted tps p50, acceptance rate.

    With `group_by` (e.g. "workload" or "mode") a dict of summaries keyed by that
    field is returned instead. Acceptance rate is sum(draft_n_accepted)/sum(draft_n)
    over records whose timings carry both, and None when no record does.
    """
    if group_by:
        groups: dict = {}
        for r in records:
            groups.setdefault(r.get(group_by), []).append(r)
        return {k: summarise(v) for k, v in groups.items()}

    ttfts = [r["ttft_s"] for r in records if r.get("ttft_s") is not None]
    ptps = [v for v in (perceived_tps(r) for r in records) if v is not None]
    ttps = [v for v in (total_tps(r) for r in records) if v is not None]
    pred = [t["predicted_per_second"] for t in (r.get("timings") or {} for r in records)
            if isinstance(t.get("predicted_per_second"), (int, float))]
    draft_n = draft_acc = 0
    draft_seen = False
    for r in records:
        t = r.get("timings") or {}
        if isinstance(t.get("draft_n"), (int, float)) and isinstance(t.get("draft_n_accepted"), (int, float)):
            draft_seen = True
            draft_n += t["draft_n"]
            draft_acc += t["draft_n_accepted"]
    acceptance = (draft_acc / draft_n) if draft_seen and draft_n > 0 else None
    shas = {r.get("sha256") for r in records if r.get("sha256")}

    def _r(v, nd=3):
        return None if v is None else round(v, nd)

    return {
        "n": len(records),
        "n_no_content": sum(1 for r in records if r.get("ttft_s") is None),
        "ttft_p50_s": _r(percentile(ttfts, 50), 4),
        "ttft_p90_s": _r(percentile(ttfts, 90), 4),
        "perceived_tps_p50": _r(percentile(ptps, 50), 1),
        "perceived_tps_p90": _r(percentile(ptps, 90), 1),
        "total_tps_p50": _r(percentile(ttps, 50), 1),
        "server_predicted_tps_p50": _r(percentile(pred, 50), 1),
        "acceptance_rate": _r(acceptance, 4),
        "draft_n": draft_n if draft_seen else None,
        "draft_n_accepted": draft_acc if draft_seen else None,
        "distinct_outputs": len(shas),
    }


def load_jsonl(path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
