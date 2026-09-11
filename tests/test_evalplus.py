"""EvalPlus (HumanEval+ / MBPP+): field mapping, program assembly, sandbox runner.

Fixture items mirror the real HF rows (field names verified 2026-09-10, see the
module docstring of lib/evals/evalplus.py). The runner tests execute a tiny
plus-style harness under the current interpreter; nothing downloads.
"""
import json
import sys
import textwrap
import types

import pytest

from lib.evals.evalplus import (DEFAULT_MAX_TOKENS, EvalPlusEval, _as_list,
                                build_humaneval_plus_program, build_mbpp_program,
                                mbpp_entry_point)

# --- fixture items in the exact HF field shape ---

_PLUS_HARNESS_HEAD = textwrap.dedent("""\
    import numpy as np

    def is_floats(x) -> bool:
        if isinstance(x, float):
            return True
        if isinstance(x, (list, tuple)):
            return all(isinstance(i, float) for i in x)
        if isinstance(x, np.ndarray):
            return x.dtype == np.float64 or x.dtype == np.float32
        return False


    def assertion(out, exp, atol):
        exact_match = out == exp
        if atol == 0 and is_floats(exp):
            atol = 1e-6
        if not exact_match and atol != 0:
            assert np.allclose(out, exp, rtol=1e-07, atol=atol)
        else:
            assert exact_match
    """)

HE_ITEM = {
    "task_id": "HumanEval/0",
    "prompt": ("def add_one(x: int) -> int:\n"
               "    \"\"\"Return x plus one.\n    >>> add_one(1)\n    2\n    \"\"\"\n"),
    "canonical_solution": "    return x + 1\n",
    "entry_point": "add_one",
    "test": _PLUS_HARNESS_HEAD + textwrap.dedent("""\

        def check(candidate):
            inputs = [[1], [2], [-5], [10**6]]
            results = [2, 3, -4, 1000001]
            for i, (inp, exp) in enumerate(zip(inputs, results)):
                assertion(candidate(*inp), exp, 0)
        """),
}

MBPP_ITEM = {
    "task_id": "2",
    "code": "\ndef similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))\n",
    "prompt": "Write a function to find the shared elements from the given two lists.",
    "source_file": "Benchmark Questions Verification V2.ipynb",
    "test_imports": [],
    "test_list": [
        "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
        "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
    ],
    "test": _PLUS_HARNESS_HEAD + textwrap.dedent("""\

        inputs = [[(3, 4, 5, 6), (5, 7, 4, 10)], [(1, 2, 3, 4), (5, 4, 3, 7)], [(), ()]]
        results = [(4, 5), (3, 4), ()]
        for i, (inp, exp) in enumerate(zip(inputs, results)):
            assertion(set(similar_elements(*inp)), set(exp), 0)
        """),
}


# --- field mapping ---

def test_as_list_accepts_list_and_repr_string():
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list("['assert f(1) == 2', 'assert f(2) == 3']") == ["assert f(1) == 2", "assert f(2) == 3"]
    assert _as_list("[]") == []
    assert _as_list(None) == []
    assert _as_list("import math") == ["import math"]  # not a repr: kept verbatim


def test_mbpp_entry_point_from_harness_then_fallbacks():
    assert mbpp_entry_point(MBPP_ITEM) == "similar_elements"
    no_harness = {**MBPP_ITEM, "test": "inputs = []"}
    assert mbpp_entry_point(no_harness) == "similar_elements"  # from test_list
    only_code = {"code": "import re\ndef helper(x):\n  pass\ndef main_fn(a):\n  pass\n",
                 "test": "", "test_list": []}
    assert mbpp_entry_point(only_code) == "main_fn"
    assert mbpp_entry_point({"test": "", "test_list": [], "code": ""}) is None


def test_mbpp_messages_show_task_and_first_assert():
    ev = EvalPlusEval(client=None, variant="mbpp_plus")
    msgs = ev.messages(MBPP_ITEM)
    assert msgs[0]["role"] == "system"
    assert MBPP_ITEM["prompt"] in msgs[1]["content"]
    assert MBPP_ITEM["test_list"][0] in msgs[1]["content"]
    assert MBPP_ITEM["test_list"][1] not in msgs[1]["content"]


def test_humaneval_plus_prompt_shape_matches_humaneval():
    from lib.evals.humaneval import _build_messages
    ev = EvalPlusEval(client=None, variant="humaneval_plus")
    assert ev.messages(HE_ITEM) == _build_messages(HE_ITEM["prompt"])
    assert ev.dataset == "evalplus/humanevalplus"
    assert EvalPlusEval(client=None, variant="mbpp_plus").dataset == "evalplus/mbppplus"


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        EvalPlusEval(client=None, variant="mbpp")


# --- program assembly ---

def test_build_mbpp_program_strips_fence_and_appends_harness():
    resp = "```python\ndef similar_elements(a, b):\n    return tuple(set(a) & set(b))\n```"
    prog = build_mbpp_program({**MBPP_ITEM, "test_imports": ["import math"]}, resp)
    assert prog.startswith("import math\n")
    assert "```" not in prog
    assert "def similar_elements(a, b):" in prog
    assert prog.rstrip().endswith("assertion(set(similar_elements(*inp)), set(exp), 0)")


def test_build_humaneval_plus_program_calls_check():
    prog = build_humaneval_plus_program(HE_ITEM, "    return x + 1")
    assert prog.startswith(HE_ITEM["prompt"])
    assert prog.rstrip().endswith("check(add_one)")


# --- runner (real subprocess under sys.executable, which has numpy) ---

def test_run_one_humaneval_plus_pass_and_fail():
    ev = EvalPlusEval(client=None, variant="humaneval_plus", exec_timeout=20)
    ok, err = ev.run_one(HE_ITEM, "    return x + 1")
    assert ok is True and err == ""
    ok, err = ev.run_one(HE_ITEM, "    return x + 2")
    assert ok is False and "AssertionError" in err


def test_run_one_mbpp_plus_pass_and_fail():
    ev = EvalPlusEval(client=None, variant="mbpp_plus", exec_timeout=20)
    good = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\n"
    ok, err = ev.run_one(MBPP_ITEM, good)
    assert ok is True and err == ""
    # passes the first original assert but not the plus edge case (empty tuples)
    bad = ("def similar_elements(a, b):\n"
           "    if not a: return (1,)\n"
           "    return tuple(set(a) & set(b))\n")
    ok, err = ev.run_one(MBPP_ITEM, bad)
    assert ok is False and "AssertionError" in err
    ok, err = ev.run_one(MBPP_ITEM, "def wrong_name(a, b): return ()")
    assert ok is False and "NameError" in err


# --- eval loop with fake client + fake datasets ---

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


def _patch_datasets(monkeypatch, items):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: list(items)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_eval_loop_humaneval_plus(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch, [HE_ITEM, {**HE_ITEM, "task_id": "HumanEval/1"}])
    client = _Client(["    return x + 1", "    return x - 1"], [12, DEFAULT_MAX_TOKENS])
    res = EvalPlusEval(client, variant="humaneval_plus", results_dir=tmp_path).evaluate()
    assert res["metric"] == "pass@1" and res["score"] == 50.0
    assert res["correct"] == res["passed"] == 1 and res["total"] == 2
    assert res["capped_count"] == 1 and res["dataset"] == "evalplus/humanevalplus"
    # standing token metrics come from token_summary (2026-09-11: were null in the first night's detail.json)
    assert res["tokens_recorded"] == 2 and res["capped_rate"] == 0.5
    assert res["tokens_per_correct"] == 12 + DEFAULT_MAX_TOKENS and res["completion_tokens_total"] == 12 + DEFAULT_MAX_TOKENS
    assert res["errors"][0]["task_id"] == "HumanEval/1"
    assert client.calls[0][1] == {"max_tokens": DEFAULT_MAX_TOKENS, "preserve_indent": True}
    prog = json.loads((tmp_path / "humaneval_plus_progress.json").read_text())["completed"]
    assert prog["0"]["correct"] is True and prog["1"]["capped"] is True
    assert (tmp_path / "humaneval_plus_detail.json").exists()
    assert len((tmp_path / "humaneval_plus_tokens.jsonl").read_text().splitlines()) == 2


def test_eval_loop_mbpp_plus_resumes(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch, [MBPP_ITEM])
    good = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\n"
    c1 = _Client([good], [40])
    r1 = EvalPlusEval(c1, variant="mbpp_plus", results_dir=tmp_path).evaluate()
    assert r1["score"] == 100.0 and r1["total"] == 1
    c2 = _Client([], [])
    r2 = EvalPlusEval(c2, variant="mbpp_plus", results_dir=tmp_path).evaluate()
    assert c2.calls == [] and r2["correct"] == 1
    assert (tmp_path / "mbpp_plus_progress.json").exists()
