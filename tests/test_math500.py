"""MATH-500: boxed extraction, grading equivalence, capped flag. Fake client, no network."""
import json
import sys
import types

import pytest

from lib.evals.math500 import (DEFAULT_MAX_TOKENS, Math500Eval, extract_boxed,
                               is_equivalent, normalize_answer)


# --- extract_boxed ---

def test_extract_simple():
    assert extract_boxed("so the answer is \\boxed{42}.") == "42"


def test_extract_last_boxed_wins():
    assert extract_boxed("\\boxed{1} ... actually \\boxed{7}") == "7"


def test_extract_nested_braces():
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed("\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}") == \
        "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert extract_boxed("x \\boxed{\\sqrt{a^{2}+b^{2}}} y") == "\\sqrt{a^{2}+b^{2}}"


def test_extract_fbox_and_spaces():
    assert extract_boxed("\\fbox{ 9 }") == "9"
    assert extract_boxed("\\boxed {12}") == "12"


def test_extract_none_when_missing_or_unclosed():
    assert extract_boxed("the answer is 42") is None
    assert extract_boxed("") is None
    assert extract_boxed("\\boxed{\\frac{1}{2") is None  # truncated completion


def test_extract_after_think_tag():
    assert extract_boxed("<think>\\boxed{1}</think>\\boxed{2}") == "2"


# --- grading ---

@pytest.mark.parametrize("pred,gold,ok", [
    ("\\frac{1}{2}", "0.5", True),                 # numeric via grader
    ("\\dfrac{3}{4}", "\\frac{3}{4}", True),        # normalised string fallback
    ("x^2 + 1", "1 + x^{2}", True),                 # symbolic via grader
    ("7", "8", False),
    (None, "8", False),
])
def test_is_equivalent(pred, gold, ok):
    assert is_equivalent(pred, gold) is ok


def test_normalize_answer():
    assert normalize_answer("\\text{Evelyn}") == "Evelyn"
    assert normalize_answer("\\left( 3, \\frac{\\pi}{2} \\right)") == "(3,\\frac{\\pi}{2})"
    assert normalize_answer("90^\\circ") == "90"


# --- eval loop ---

class _Client:
    def __init__(self, answers, tokens):
        self.answers, self.tokens, self.calls = list(answers), list(tokens), []
        self.last_completion_tokens = None
        self.reasoning_fallback_count = 0

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        i = len(self.calls) - 1
        self.last_completion_tokens = self.tokens[i]
        return self.answers[i]


_ITEMS = [
    {"problem": "p1", "solution": "s", "answer": "\\frac{1}{2}", "subject": "Algebra",
     "level": 1, "unique_id": "test/algebra/1.json"},
    {"problem": "p2", "solution": "s", "answer": "3", "subject": "Algebra",
     "level": 2, "unique_id": "test/algebra/2.json"},
    {"problem": "p3", "solution": "s", "answer": "10", "subject": "Number Theory",
     "level": 3, "unique_id": "test/nt/3.json"},
]


def _patch_datasets(monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: list(_ITEMS)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_eval_scores_and_flags_capped(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch)
    client = _Client(["\\boxed{0.5}", "\\boxed{4}", "ran out of budget \\boxed{1"],
                     [100, 200, DEFAULT_MAX_TOKENS])
    res = Math500Eval(client, results_dir=tmp_path).evaluate()
    assert res["metric"] == "acc"
    assert res["correct"] == 1 and res["total"] == 3
    assert res["score"] == pytest.approx(33.33, abs=0.01)
    assert res["parse_failures"] == 1
    assert res["capped_count"] == 1 and res["max_tokens"] == DEFAULT_MAX_TOKENS == 2048
    assert res["completion_tokens_total"] == 100 + 200 + 2048
    assert client.calls[0][1]["max_tokens"] == 2048
    assert "\\boxed" in client.calls[0][0][0]["content"]
    prog = json.loads((tmp_path / "math500_progress.json").read_text())["completed"]
    assert prog["2"]["capped"] is True and prog["2"]["predicted"] is None
    assert prog["0"]["capped"] is False and prog["0"]["completion_tokens"] == 100
    assert prog["1"]["expected"] == "3"
    assert (tmp_path / "math500_detail.json").exists()
    assert len((tmp_path / "math500_tokens.jsonl").read_text().splitlines()) == 3


def test_eval_resumes_and_custom_budget(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch)
    c1 = _Client(["\\boxed{1/2}", "\\boxed{3}", "\\boxed{10}"], [10, 10, 505])
    r1 = Math500Eval(c1, results_dir=tmp_path, max_tokens=512).evaluate()
    assert r1["correct"] == 3 and r1["capped_count"] == 1  # 505 >= 512 - CAP_SLACK(8)
    c2 = _Client([], [])
    r2 = Math500Eval(c2, results_dir=tmp_path, max_tokens=512).evaluate()
    assert c2.calls == [] and r2["correct"] == 3 and r2["total"] == 3


def test_limit(monkeypatch):
    _patch_datasets(monkeypatch)
    client = _Client(["\\boxed{2}"], [1])
    res = Math500Eval(client, limit=1).evaluate()
    assert res["total"] == 1 and res["correct"] == 0 and len(client.calls) == 1
