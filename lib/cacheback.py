"""Cache-only (draft-free) speculative-decoding invariants.

Per ADR-0003, every "what does this number mean" decision is a deterministic, tested
pure function here; the GPU driver (scripts/bench_cacheback.py) only orchestrates.

Greedy spec-decode is LOSSLESS by construction: the verify loop accepts a draft token
only when it equals the target model's greedy argmax, and always appends the model's own
argmax as a correction. So the emitted sequence is identical to plain greedy decode,
whatever the drafter proposes. That identity is what assert_lossless checks empirically.

Speedup can never exceed MAT (mean accepted tokens / step): each step is ONE target
forward pass advancing MAT tokens, and a spec forward (longer input) costs >= an AR
forward, so realized speedup <= MAT. A measured speedup above MAT means an instrumentation
bug, not a faster decoder (cf. t036).
"""
from __future__ import annotations


class LosslessnessError(AssertionError):
    pass


class SpeedupSanityError(AssertionError):
    pass


def greedy_accept(draft: list[int], target_preds: list[int]) -> tuple[list[int], int]:
    """Accept the longest prefix of `draft` matching the target's greedy preds, then
    append one correction/bonus token. `target_preds` has len(draft)+1 entries: prediction
    i is the greedy argmax at the position that consumes draft[:i]. Returns
    (accepted_tokens, n_advanced); n_advanced >= 1 always."""
    assert len(target_preds) == len(draft) + 1, "need one prediction per draft slot + bonus"
    accepted: list[int] = []
    for i, tok in enumerate(draft):
        if tok == target_preds[i]:
            accepted.append(tok)
        else:
            break
    accepted.append(target_preds[len(accepted)])  # correction (or final bonus)
    return accepted, len(accepted)


def mean_accepted_tokens(advances: list[int]) -> float:
    if not advances:
        return 0.0
    return sum(advances) / len(advances)


def assert_lossless(ar_ids: list[int], spec_ids: list[int]) -> None:
    if ar_ids != spec_ids:
        n = min(len(ar_ids), len(spec_ids))
        first = next((i for i in range(n) if ar_ids[i] != spec_ids[i]), n)
        raise LosslessnessError(
            f"spec output diverged from AR at index {first} "
            f"(len AR={len(ar_ids)}, spec={len(spec_ids)})"
        )


def assert_speedup_sane(speedup: float, mat: float, eps: float = 0.05) -> None:
    if speedup > mat + eps:
        raise SpeedupSanityError(
            f"speedup {speedup:.3f} > MAT {mat:.3f} (+{eps}): physically impossible"
        )
