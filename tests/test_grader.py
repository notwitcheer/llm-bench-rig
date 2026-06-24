import pytest
from lib.qwen_math_grader import extract_answer, grade


def test_extract_boxed():
    assert extract_answer("...therefore \\boxed{42}.") == "42"


def test_extract_last_boxed_wins():
    assert extract_answer("\\boxed{1} then \\boxed{7}") == "7"


def test_extract_number_fallback():
    assert extract_answer("the answer is 13") == "13"


@pytest.mark.parametrize("pred,gold,ok", [
    ("42", "42", True),
    ("\\frac{1}{2}", "0.5", True),     # symbolic equality
    ("7", "8", False),
])
def test_grade(pred, gold, ok):
    assert grade(pred, gold) is ok
