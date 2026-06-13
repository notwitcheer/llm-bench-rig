"""Speculative-decoding metrics: parse vLLM /metrics counters and compute the
invariants of a spec-decode comparison (acceptance length, acceptance rate, speedup).

Per ADR-0003 the invariants live here as deterministic, tested helpers — the driver
(scripts/bench_specdecode.py) only orchestrates servers and HTTP; every "what does
this number mean" decision is a pure function below.

The three-way (MTP vs EAGLE3 vs DFlash) turns on TWO axes, and they can disagree:
  - acceptance_length: mean tokens advanced per target forward pass. Higher = the
    drafter guessed more correct tokens per verify. DFlash's pitch is a FLAT high
    length (~8-9) where MTP/EAGLE decay with draft depth.
  - decode tok/s speedup vs the no-spec baseline: the realized wall-clock win, which
    also pays the drafter's own forward cost. A big drafter (DFlash) can post a higher
    acceptance length yet a smaller speedup if its draft pass is expensive — which is
    exactly the local-model crossover this bench measures.

vLLM metric names drift across versions (v0 vs v1, `_total` suffix, label braces), so
the parser matches loosely on the `spec_decode` family rather than exact strings.
"""
from __future__ import annotations

import re

# Match ONLY the three exact spec-decode counters, with an optional `_total` suffix and
# optional Prometheus labels. Crucially this must NOT match the look-alike series that vLLM
# emits alongside them and that collide by prefix:
#   - `num_accepted_tokens_per_pos{_total}` — a per-POSITION histogram (often 0.0); matching it
#     in place of the real total silently zeroes acceptance.
#   - `..._created` — a Prometheus creation TIMESTAMP (~1.78e9), not a count.
# The trailing `(?:_total)?(?:\{...\})?\s` anchors the name exactly, so `_per_pos...` / `_created`
# (which have more name chars before the value) never match.
_COUNTER = re.compile(
    r"^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)"
    r"(?:_total)?(?:\{[^}]*\})?\s+(?P<val>[-+0-9.eE]+)\s*$"
)


def parse_spec_metrics(metrics_text: str) -> dict[str, float]:
    """Extract {num_drafts, num_draft_tokens, num_accepted_tokens} from a vLLM /metrics dump.

    Counters are cumulative — diff two snapshots with `delta`. Only the exact `_total`
    counters are read; the per-position histogram and `_created` timestamps are excluded
    (matching them by prefix was a real bug — it non-deterministically zeroed acceptance).
    """
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        m = _COUNTER.match(line.strip())
        if m:
            out[m.group(1)] = float(m.group("val"))
    return out


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Counter difference over a measurement window (after - before), clamped at 0.

    Cumulative counters can only grow; a negative delta means the server reset
    (restart between configs) — clamp to 0 rather than emit a nonsense rate.
    """
    keys = set(before) | set(after)
    return {k: max(after.get(k, 0.0) - before.get(k, 0.0), 0.0) for k in keys}


def acceptance_length(accepted_tokens: float, num_drafts: float) -> float | None:
    """Mean tokens advanced per target forward pass = accepted/drafts + 1.

    The +1 is the bonus token the target itself produces on every verify step (always
    "accepted"). A no-spec run advances exactly 1 token/step, so this is directly
    comparable to a baseline of 1.0. Returns None if no draft steps were recorded.
    """
    if not num_drafts or num_drafts <= 0:
        return None
    return round(accepted_tokens / num_drafts + 1.0, 3)


def acceptance_rate(accepted_tokens: float, draft_tokens: float) -> float | None:
    """Per-token accept probability = accepted/proposed, in [0,1]. None if nothing drafted."""
    if not draft_tokens or draft_tokens <= 0:
        return None
    return round(accepted_tokens / draft_tokens, 4)


def speedup(spec_tps: float, base_tps: float) -> float | None:
    """Realized decode speedup vs the no-spec baseline. None if baseline is missing/zero."""
    if not base_tps or base_tps <= 0:
        return None
    return round(spec_tps / base_tps, 3)


def summarize_acceptance(metrics_delta: dict[str, float]) -> dict[str, float | None]:
    """Acceptance length + rate from a windowed counter delta. The headline pair."""
    acc = metrics_delta.get("num_accepted_tokens", 0.0)
    drafts = metrics_delta.get("num_drafts", 0.0)
    dtoks = metrics_delta.get("num_draft_tokens", 0.0)
    return {
        "acceptance_length": acceptance_length(acc, drafts),
        "acceptance_rate": acceptance_rate(acc, dtoks),
        "accepted_tokens": acc or None,
        "draft_tokens": dtoks or None,
        "num_drafts": drafts or None,
    }
