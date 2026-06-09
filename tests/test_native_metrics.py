from lib.agentic.native.metrics import score_run, agentic_score


def test_per_run_success_and_efficiency():
    # solved in optimal calls, no bad calls, no stall
    s = score_run(success=True, n_tool_calls=2, opt_calls=2, bad_calls=0, stalled=False)
    assert s["success"] == 1.0 and s["tool_eff"] == 1.0 and s["stable"] == 1.0


def test_extra_calls_reduce_efficiency():
    s = score_run(success=True, n_tool_calls=4, opt_calls=2, bad_calls=1, stalled=False)
    assert s["tool_eff"] < 1.0


def test_agentic_score_weights_sum_100():
    runs = [{"success": 1.0, "tool_eff": 1.0, "stable": 1.0}] * 4
    out = agentic_score(runs, tokens_per_task=500.0)
    # all-perfect success/eff/stability -> 0.50+0.20+0.15 of 100 = 85 before token axis
    assert 84.9 <= out["score"] <= 100.0
    assert out["task_success_pct"] == 100.0
