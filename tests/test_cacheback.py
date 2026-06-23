import pytest
from lib.cacheback import (
    greedy_accept, mean_accepted_tokens,
    assert_lossless, assert_speedup_sane,
    LosslessnessError, SpeedupSanityError,
)


def test_all_draft_tokens_accepted_plus_bonus():
    # draft fully matches the target's greedy preds -> accept all 3 + 1 bonus
    accepted, n = greedy_accept([10, 11, 12], [10, 11, 12, 13])
    assert accepted == [10, 11, 12, 13]
    assert n == 4


def test_first_mismatch_truncates_and_corrects():
    # draft[1] disagrees with target -> accept draft[0], then correction token 99
    accepted, n = greedy_accept([10, 11, 12], [10, 99, 12, 13])
    assert accepted == [10, 99]
    assert n == 2


def test_empty_draft_advances_one():
    # AR case: no draft, just the single greedy token
    accepted, n = greedy_accept([], [42])
    assert accepted == [42]
    assert n == 1


def test_mean_accepted_tokens():
    assert mean_accepted_tokens([1, 1, 4, 2]) == 2.0


def test_lossless_passes_and_fails():
    assert_lossless([1, 2, 3], [1, 2, 3])  # no raise
    with pytest.raises(LosslessnessError):
        assert_lossless([1, 2, 3], [1, 2, 4])


def test_speedup_sanity():
    assert_speedup_sane(2.3, 2.42)          # ok, below MAT
    assert_speedup_sane(2.45, 2.42)         # ok, within eps
    with pytest.raises(SpeedupSanityError):
        assert_speedup_sane(3.0, 2.42)      # impossible: faster than tokens/forward


# --- Task 2: n-gram drafters ---
from lib.cacheback import pld_propose, NGramLRU


def test_pld_proposes_recent_continuation():
    # "1 2 3" appeared after the last "1"; propose its continuation
    seq = [1, 2, 3, 9, 1]
    assert pld_propose(seq, n=1, max_draft=2) == [2, 3]


def test_pld_no_match_returns_empty():
    assert pld_propose([1, 2, 3], n=1, max_draft=3) == []  # last token "3" never seen earlier


def test_pld_prefers_most_recent_occurrence():
    seq = [7, 5, 7, 8, 7]   # last "7"; most recent earlier "7" is at idx2 -> followed by 8
    assert pld_propose(seq, n=1, max_draft=1) == [8]


def test_ngram_lru_eviction_and_recency():
    t = NGramLRU(capacity=2)
    t.update((1,), (10,)); t.update((2,), (20,)); t.update((3,), (30,))  # evicts (1,)
    assert t.lookup((1,)) == []
    assert t.lookup((3,)) == [(30,)]
    t.update((2,), (21,))                       # most-recent first
    assert t.lookup((2,))[0] == (21,)


# --- Task 4: spec-decode loop (torch-free) ---
from lib.cacheback import spec_decode_loop


def _cyclic_forward_argmax(seq, draft):
    """Stub target model whose 'true' sequence is period-3: token at position p == p % 3.
    Returns greedy argmax for the len(draft)+1 positions after seq (independent of draft
    values, so the verified output is deterministically period-3 whatever the drafter does)."""
    base = len(seq) - 1
    return [(base + i + 1) % 3 for i in range(len(draft) + 1)]


def test_loop_pld_is_lossless_and_faster_than_ar():
    seed = [0, 1, 2]
    ar, adv_ar = spec_decode_loop(
        _cyclic_forward_argmax, seed, propose=lambda s: [], max_new=30)
    pld, adv_pld = spec_decode_loop(
        _cyclic_forward_argmax, seed, propose=lambda s: pld_propose(s, 1, 3), max_new=30)
    m = min(len(ar), len(pld))
    assert m >= 30
    assert ar[:m] == pld[:m]               # lossless: identical output whatever the drafter
    assert all(a == 1 for a in adv_ar)     # AR advances exactly 1 per forward pass
    assert max(adv_pld) > 1                 # PLD accepts multiple tokens in some steps
    assert len(adv_pld) < len(adv_ar)      # => fewer forward passes (real acceleration)
