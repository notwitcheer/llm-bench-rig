"""Per-run scoring + Agentic Score aggregation. Token efficiency is scored RELATIVELY
at publish time (vs the foil); within one model's run we report raw tokens/task and a
provisional full-credit token axis (the cross-model delta is computed in the report)."""
WEIGHTS = {"success": 0.50, "tool_eff": 0.20, "token_eff": 0.15, "stable": 0.15}


def score_run(success: bool, n_tool_calls: int, opt_calls: int, bad_calls: int, stalled: bool) -> dict:
    eff = 1.0
    if n_tool_calls > 0:
        eff = min(1.0, opt_calls / n_tool_calls)            # extra calls cost
    eff = max(0.0, eff - 0.25 * bad_calls)                  # malformed/failed calls cost
    return {"success": 1.0 if success else 0.0, "tool_eff": round(eff, 3),
            "stable": 0.0 if stalled else 1.0}


def agentic_score(runs: list, tokens_per_task: float, token_eff: float = 1.0) -> dict:
    n = max(1, len(runs))
    succ = sum(r["success"] for r in runs) / n
    eff = sum(r["tool_eff"] for r in runs) / n
    stable = sum(r["stable"] for r in runs) / n
    score = 100 * (WEIGHTS["success"] * succ + WEIGHTS["tool_eff"] * eff
                   + WEIGHTS["token_eff"] * token_eff + WEIGHTS["stable"] * stable)
    return {"score": round(score, 2), "task_success_pct": round(100 * succ, 1),
            "tool_eff": round(eff, 3), "stable_pct": round(100 * stable, 1),
            "tokens_per_task": round(tokens_per_task, 1)}


def longctx_summary(runs: list) -> dict:
    """Long-context is scored separately from the main Agentic Score: just task success
    at this tier (input size is a property of the tier, not the model)."""
    n = len(runs)
    succ = sum(1 for r in runs if r.get("success")) / n if n else 0.0
    return {"n": n, "success_pct": round(100 * succ, 1)}
