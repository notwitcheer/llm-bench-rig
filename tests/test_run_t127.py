"""t127 orchestration, offline: plan_legs / run_leg / expected_runs / project_hours.
GPU serving (main(), --probe's real run) is NOT exercised here — see run_t127.py's
module docstring. Uses the ScriptedClient pattern from tests/test_native_agent_loop.py
(copied, not cross-imported, per that test file's own convention)."""
from scripts.run_t127 import expected_runs, plan_legs, project_hours, reasoning_tokens_ok, run_leg
from lib.agentic.native.tasks_t127 import T127_TASKS


class ScriptedClient:
    """Returns one pre-baked assistant message then stops (no tool calls)."""
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


def _task(task_id: str) -> dict:
    return next(t for t in T127_TASKS if t["id"] == task_id)


_CFG = {
    "t127": {
        "models": [
            {"slug": "LFM2.5-2.6B", "gguf": "~/t127-lfm/model/LFM2.5-2.6B-Q8_0.gguf"},
            {"slug": "Qwen3.5-9B", "gguf": "~/models/qwen3.5-9b-q8_0.gguf"},
            {"slug": "Gemma-4-E4B-it", "gguf": "~/models/gemma-4-e4b-it-q8_0.gguf"},
            {"slug": "Qwen3.5-4B", "gguf": "~/models/qwen3.5-4b-q8_0.gguf"},
        ],
        "n_tasks": 30, "replicates": 3, "max_steps": 12, "budget": 2048,
    }
}


def test_plan_legs_returns_five_legs():
    legs = plan_legs(_CFG)
    assert len(legs) == 5
    tuples = [(l["slug"], l["think"], l["regime"]) for l in legs]
    assert ("LFM2.5-2.6B", True, "reasoning_on") in tuples
    assert ("LFM2.5-2.6B", False, "reasoning_off") in tuples
    assert ("Qwen3.5-9B", True, "default") in tuples
    assert ("Gemma-4-E4B-it", True, "default") in tuples
    assert ("Qwen3.5-4B", True, "default") in tuples


def test_plan_legs_lfm25_appears_exactly_twice():
    legs = plan_legs(_CFG)
    lfm_legs = [l for l in legs if l["slug"] == "LFM2.5-2.6B"]
    assert len(lfm_legs) == 2
    assert {l["regime"] for l in lfm_legs} == {"reasoning_on", "reasoning_off"}


def test_plan_legs_every_leg_has_the_required_keys():
    for leg in plan_legs(_CFG):
        assert set(leg) == {"slug", "gguf", "think", "regime"}


def test_expected_runs():
    assert expected_runs(30, 3) == 450


def test_expected_runs_custom_n_legs():
    assert expected_runs(10, 2, n_legs=3) == 60


def test_project_hours():
    assert project_hours(40, 30, 3) == 40 * 450 / 3600


def test_run_leg_produces_the_right_row_count_and_keys():
    leg = {"slug": "test-model", "gguf": "/fake/model.gguf", "think": True, "regime": "default"}
    tasks = [_task("coding_factorial5"), _task("mtif_vram_json")]
    reps = 2

    # scripted answers, one per (task, rep) call in run_leg's iteration order
    # (outer task loop, inner rep loop) -- task 1 (coding_factorial5) answered
    # correctly both reps; task 2 (mtif_vram_json) correct on rep 0, wrong on rep 1,
    # so `ok` is verified to vary per-row, not just be blanket True.
    scripted_texts = [
        "the factorial of 5 is 120.",
        "the factorial of 5 is 120.",
        '{"vram_gb": 32, "doubled": 64}',
        '{"vram_gb": 32, "doubled": 999}',
    ]
    calls = {"n": 0}

    def make_client(_leg):
        text = scripted_texts[calls["n"]]
        calls["n"] += 1
        return ScriptedClient([{"role": "assistant", "content": text, "tool_calls": None}])

    rows = run_leg(leg, tasks, reps, make_client, tools=[], dispatch=None)

    assert len(rows) == len(tasks) * reps
    required_keys = {"slug", "regime", "think", "task_id", "rep", "ok", "tool_calls",
                     "bad_calls", "gen_tokens", "reasoning_tokens", "stalled"}
    for row in rows:
        assert set(row) == required_keys
        assert row["slug"] == "test-model"
        assert row["regime"] == "default"
        assert row["think"] is True

    assert [r["task_id"] for r in rows] == ["coding_factorial5", "coding_factorial5",
                                            "mtif_vram_json", "mtif_vram_json"]
    assert [r["rep"] for r in rows] == [0, 1, 0, 1]
    assert [r["ok"] for r in rows] == [True, True, True, False]
    assert calls["n"] == 4  # make_client called exactly once per (task, rep)


def _row(regime: str, reasoning_tokens: int) -> dict:
    return {"slug": "x", "regime": regime, "think": True, "task_id": "t", "rep": 0,
            "ok": True, "tool_calls": 1, "bad_calls": 0, "gen_tokens": 10,
            "reasoning_tokens": reasoning_tokens, "stalled": False}


def test_reasoning_tokens_ok_true_when_a_reasoning_on_row_has_tokens():
    rows = [_row("reasoning_off", 0), _row("reasoning_on", 0), _row("reasoning_on", 42),
            _row("default", 0)]
    assert reasoning_tokens_ok(rows) is True


def test_reasoning_tokens_ok_false_when_all_reasoning_on_rows_are_zero():
    rows = [_row("reasoning_off", 5), _row("reasoning_on", 0), _row("reasoning_on", 0),
            _row("default", 7)]
    assert reasoning_tokens_ok(rows) is False
