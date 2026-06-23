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
