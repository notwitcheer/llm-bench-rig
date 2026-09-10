"""Second-tier task wiring: registered, dispatched by _run_evals, never in q_avg."""
import json

import pytest

import lib.quality as q
from lib.board import BOARD_TASKS, QUALITY_TASKS, quality_average
from lib.ci import BOARD_TASKS as CI_BOARD_TASKS

NEW_TASKS = ("ifeval", "math500", "humaneval_plus", "mbpp_plus")


def test_new_tasks_registered_and_second_tier():
    for t in NEW_TASKS:
        assert t in q.EVAL_REGISTRY
        assert t in q.SECOND_TIER_TASKS
        assert t not in q.MC_TASKS  # generative: no MC completion-length gate


def test_board_allowlist_unchanged_and_excludes_new_tasks():
    assert QUALITY_TASKS == ("mmlu", "arc_challenge", "hellaswag", "humaneval", "gsm8k")
    assert set(CI_BOARD_TASKS) == BOARD_TASKS
    for t in NEW_TASKS + ("gpqa",):
        assert t not in BOARD_TASKS
        assert t not in CI_BOARD_TASKS


def test_quality_average_ignores_new_tasks():
    quality = {t: {"score": 80.0} for t in QUALITY_TASKS}
    base = quality_average(quality)
    quality.update({t: {"score": 5.0} for t in NEW_TASKS})
    assert quality_average(quality) == base == 80.0


def test_make_evaluator_dispatch(monkeypatch):
    seen = {}

    def mk(name):
        def fake(**kwargs):
            seen[name] = kwargs
            return object()
        return fake

    for cls in ["IFEvalEval", "Math500Eval", "EvalPlusEval"]:
        monkeypatch.setattr(q, cls, mk(cls))
    q._make_evaluator("ifeval", "c", "rd", None, limit=7)
    assert seen["IFEvalEval"] == {"client": "c", "limit": 7, "results_dir": "rd"}
    q._make_evaluator("math500", "c", "rd", None, limit=7)
    assert seen["Math500Eval"] == {"client": "c", "limit": 7, "results_dir": "rd"}
    q._make_evaluator("humaneval_plus", "c", "rd", None, limit=7)
    assert seen["EvalPlusEval"]["variant"] == "humaneval_plus"
    q._make_evaluator("mbpp_plus", "c", "rd", None, limit=7)
    assert seen["EvalPlusEval"]["variant"] == "mbpp_plus"
    with pytest.raises(ValueError):
        q._make_evaluator("mbpp", "c", "rd", None)


def test_run_evals_writes_detail_per_new_task(tmp_path, monkeypatch):
    class _Ev:
        def __init__(self, task):
            self.task = task

        def evaluate(self):
            return {"score": 42.0, "metric": f"m_{self.task}", "correct": 1, "total": 2,
                    "parse_failures": 0}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(q, "LLMClient", _Client)
    monkeypatch.setattr(q, "_make_evaluator", lambda task, *a, **k: _Ev(task))
    res = q._run_evals("http://x", "m", list(NEW_TASKS), tmp_path, None, think=False)
    assert set(res) == set(NEW_TASKS)
    for t in NEW_TASKS:
        assert res[t] == {"score": 42.0, "metric": f"m_{t}"}
        detail = json.loads((tmp_path / f"{t}_detail.json").read_text())
        assert detail["correct"] == 1 and detail["total"] == 2
