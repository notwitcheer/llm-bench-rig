import math
import pytest
from lib.em import entropy_loss_ref, repetition_rate, mean_len


def test_entropy_uniform_is_max():
    # uniform over V=4 at temp 1 -> entropy = ln(4); one generated position
    logits = [[0.0, 0.0, 0.0, 0.0]]
    assert entropy_loss_ref(logits, [1], temp=1.0) == pytest.approx(math.log(4))


def test_entropy_peaked_is_low():
    logits = [[100.0, 0.0, 0.0, 0.0]]   # near one-hot
    assert entropy_loss_ref(logits, [1], temp=1.0) == pytest.approx(0.0, abs=1e-3)


def test_mask_excludes_prompt_positions():
    # two positions, only the second is "generated"; first is high-entropy, must be ignored
    logits = [[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]
    assert entropy_loss_ref(logits, [0, 1], temp=1.0) == pytest.approx(0.0, abs=1e-3)


def test_temp_sharpens():
    # lower softmax temp -> sharper -> lower entropy than temp=1 for the same logits
    logits = [[2.0, 1.0, 0.0, 0.0]]
    assert entropy_loss_ref(logits, [1], temp=0.5) < entropy_loss_ref(logits, [1], temp=1.0)


def test_repetition_rate():
    assert repetition_rate([1, 2, 3, 4, 5], n=2) == pytest.approx(0.0)      # all bigrams unique
    assert repetition_rate([7, 7, 7, 7], n=2) == pytest.approx(1 - 1 / 3)   # (7,7)x3 -> 1 unique/3


def test_mean_len():
    assert mean_len([[1, 2, 3], [1]]) == pytest.approx(2.0)
