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


# --- n-gram drafters (proposers) ---
from collections import OrderedDict, deque


def pld_propose(seq: list[int], n: int, max_draft: int) -> list[int]:
    """Prompt-lookup draft: find the latest earlier occurrence of seq[-n:] and return the
    up-to-max_draft tokens that followed it. Searches the full sequence-so-far (prompt +
    generated), so it is 'dynamic' over generated text like Cacheback's dynamic table."""
    if len(seq) <= n:
        return []
    key = seq[-n:]
    # scan right-to-left over candidate start positions (most recent match wins)
    for start in range(len(seq) - n - 1, -1, -1):
        if seq[start:start + n] == key:
            cont = seq[start + n: start + n + max_draft]
            return list(cont)
    return []


class NGramLRU:
    """Capacity-bounded LRU map leader-tuple -> most-recently-seen follower-tuples."""
    def __init__(self, capacity: int, followers_per_leader: int = 4):
        self.capacity = capacity
        self.fpl = followers_per_leader
        self._d: "OrderedDict[tuple, deque]" = OrderedDict()

    def update(self, leader: tuple, followers: tuple) -> None:
        if leader in self._d:
            self._d.move_to_end(leader)
        else:
            self._d[leader] = deque(maxlen=self.fpl)
            if len(self._d) > self.capacity:
                self._d.popitem(last=False)  # evict least-recently-used
        self._d[leader].appendleft(followers)

    def lookup(self, leader: tuple) -> list:
        if leader not in self._d:
            return []
        return list(self._d[leader])  # most-recent first


# --- workload loading ---
import json


def load_workload(path: str) -> list[dict]:
    """Load a workload jsonl (schema: {id, workload, prompt}); validate non-empty + keys."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert rows and all({"id", "workload", "prompt"} <= r.keys() for r in rows)
    return rows
