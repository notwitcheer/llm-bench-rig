"""Tests that _make_evaluator correctly forwards limit and mmlu_limit caps."""

import lib.quality as q


def test_make_evaluator_caps(monkeypatch):
    seen = {}

    def mk(name):
        def fake(**kwargs):
            seen[name] = kwargs
            return object()
        return fake

    for cls in ["MMLUEval", "ARCEval", "HellaSwagEval", "GSM8KEval", "HumanEvalEval"]:
        monkeypatch.setattr(q, cls, mk(cls))

    # MMLU: should receive mmlu_limit (not limit)
    q._make_evaluator("mmlu", None, None, 0.5, limit=100, mmlu_limit=2)
    # arc_challenge: should receive limit (flat cap)
    q._make_evaluator("arc_challenge", None, None, None, limit=100, mmlu_limit=2)
    # hellaswag: should receive limit
    q._make_evaluator("hellaswag", None, None, 0.5, limit=100, mmlu_limit=2)
    # gsm8k: should receive limit
    q._make_evaluator("gsm8k", None, None, None, limit=100, mmlu_limit=2)
    # humaneval: should receive limit
    q._make_evaluator("humaneval", None, None, None, limit=100, mmlu_limit=2)

    # MMLU uses mmlu_limit as its limit kwarg, not the flat limit
    assert seen["MMLUEval"]["limit"] == 2
    assert seen["MMLUEval"]["sample"] == 0.5

    # Non-MMLU tasks use the flat limit
    assert seen["ARCEval"]["limit"] == 100
    assert seen["HellaSwagEval"]["limit"] == 100
    assert seen["HellaSwagEval"]["sample"] == 0.5
    assert seen["GSM8KEval"]["limit"] == 100
    assert seen["HumanEvalEval"]["limit"] == 100


def test_make_evaluator_none_caps(monkeypatch):
    """When both limits are None, evaluators are still constructed without errors."""
    seen = {}

    def mk(name):
        def fake(**kwargs):
            seen[name] = kwargs
            return object()
        return fake

    for cls in ["MMLUEval", "ARCEval", "HellaSwagEval", "GSM8KEval", "HumanEvalEval"]:
        monkeypatch.setattr(q, cls, mk(cls))

    q._make_evaluator("mmlu", None, None, None, limit=None, mmlu_limit=None)
    q._make_evaluator("gsm8k", None, None, None, limit=None, mmlu_limit=None)

    assert seen["MMLUEval"]["limit"] is None
    assert seen["GSM8KEval"]["limit"] is None


def test_make_evaluator_unknown_task(monkeypatch):
    """Unknown task names raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown eval task"):
        q._make_evaluator("bogus_task", None, None, None, limit=None, mmlu_limit=None)
