"""t127 analysis: turn the sweep's result rows (results/t127/results.jsonl, one dict
per task x rep x leg -- see scripts/run_t127.py's `run_leg`) into the bench's
statistics. Report/chart writing needs real GPU results and lives elsewhere (a later,
GATED task); this module is pure and offline-testable: `analyze(rows) -> dict`.

Reuses, does NOT reimplement:
  - `lib.agentic.phaseb.wilson_ci`          -- per-leg / per-tier success CIs.
  - `lib.agentic.native.paired.mcnemar`     -- paired significance tests.
  - `lib.agentic.native.tasks_t127.T127_TASKS` -- task_id -> tier / opt_calls maps.
  - `lib.agentic.native.metrics.score_run`  -- per-row tool_eff (extra-calls +
    bad-calls cost), folded into the efficiency block's `mean_tool_eff`.

A "leg" is a unique (slug, regime) pair, keyed in the output as "<slug>|<regime>".

Majority vote (for the paired tests): each task_id's replicate `ok` outcomes are
reduced to ONE boolean per leg via majority vote (more True than False among its
reps) BEFORE pairing -- McNemar needs one outcome per matched unit, not per rep. An
exact tie (possible with an even rep count) resolves to False; a tie is not a pass.

Pairing: LFM2.5's `reasoning_on` leg is identified by regime (not slug) against every
`default`-regime leg (the anchors) and against the `reasoning_off` leg, if present.
Only task_ids present in BOTH legs of a pair are compared (a leg missing a task_id
entirely -- e.g. a partial/aborted run -- must not silently zero-fill it).

Efficiency ("intelligence per watt", reasoning tokens are the cost): mean gen/
reasoning/total tokens are computed PER TASK (averaged over that task's reps first),
then averaged across tasks unweighted -- this differs from a flat mean over all rows
whenever tasks have different rep counts, which real t127 legs don't but is worth
being explicit about since a bug here would silently mismeasure exactly the
metric the whole bench exists to report. `success_per_1k_tokens` uses the RAW token
total across every row of the leg (not the per-task mean) as the cost denominator,
since it represents literal total compute spent, not a per-task rate.
"""
from lib.agentic.native import metrics
from lib.agentic.native.paired import mcnemar
from lib.agentic.native.tasks_t127 import T127_TASKS
from lib.agentic.phaseb import wilson_ci

_TASK_TIER = {t["id"]: t["tier"] for t in T127_TASKS}
_TASK_OPT_CALLS = {t["id"]: t.get("opt_calls", 1) for t in T127_TASKS}


def _leg_key(slug: str, regime: str) -> str:
    return f"{slug}|{regime}"


def _group_by(rows: list, key_fn) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out


def _majority_vote(oks: list) -> bool:
    """True iff strictly more than half the reps succeeded; an exact tie -> False."""
    return sum(1 for o in oks if o) * 2 > len(oks)


def _task_votes(rows_for_leg: list) -> dict:
    """task_id -> majority-vote bool for one leg's rows."""
    by_task = _group_by(rows_for_leg, lambda r: r["task_id"])
    return {tid: _majority_vote([r["ok"] for r in trows]) for tid, trows in by_task.items()}


def _success_block(rows: list) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["ok"])
    return {
        "n": n,
        "successes": k,
        "success_rate_pct": round(100 * k / n, 1) if n else 0.0,
        "success_ci_pct": wilson_ci(k, n),
    }


def _tier_blocks(rows_for_leg: list) -> dict:
    by_tier = _group_by(rows_for_leg, lambda r: _TASK_TIER.get(r["task_id"], "unknown"))
    return {tier: _success_block(trows) for tier, trows in by_tier.items()}


def _mean_per_task(rows_for_leg: list, field: str) -> float:
    """Per-task mean of `field` across that task's reps, then unweighted mean across
    tasks. See module docstring for why (uneven rep counts) this differs from a flat
    mean over all rows."""
    by_task = _group_by(rows_for_leg, lambda r: r["task_id"])
    task_means = [sum(r[field] for r in trows) / len(trows) for trows in by_task.values()]
    return sum(task_means) / len(task_means) if task_means else 0.0


def _efficiency_block(rows_for_leg: list) -> dict:
    n = len(rows_for_leg)
    successes = sum(1 for r in rows_for_leg if r["ok"])
    mean_gen = _mean_per_task(rows_for_leg, "gen_tokens")
    mean_reasoning = _mean_per_task(rows_for_leg, "reasoning_tokens")
    total_tokens = sum(r["gen_tokens"] + r["reasoning_tokens"] for r in rows_for_leg)
    tool_eff = [
        metrics.score_run(r["ok"], r["tool_calls"], _TASK_OPT_CALLS.get(r["task_id"], 1),
                          r["bad_calls"], r["stalled"])["tool_eff"]
        for r in rows_for_leg
    ]
    return {
        "mean_gen_tokens_per_task": round(mean_gen, 2),
        "mean_reasoning_tokens_per_task": round(mean_reasoning, 2),
        "mean_total_tokens_per_task": round(mean_gen + mean_reasoning, 2),
        "success_per_1k_tokens": (successes / (total_tokens / 1000)) if total_tokens else 0.0,
        "mean_tool_calls": round(_mean_per_task(rows_for_leg, "tool_calls"), 2),
        "mean_bad_calls": round(_mean_per_task(rows_for_leg, "bad_calls"), 2),
        "mean_tool_eff": round(sum(tool_eff) / n, 3) if n else 0.0,
    }


def _leg_entry(rows_for_leg: list) -> dict:
    think_vals = {r["think"] for r in rows_for_leg}
    entry = {
        "slug": rows_for_leg[0]["slug"],
        "regime": rows_for_leg[0]["regime"],
        "think": think_vals.pop() if len(think_vals) == 1 else None,
    }
    entry.update(_success_block(rows_for_leg))
    entry["tiers"] = _tier_blocks(rows_for_leg)
    entry["efficiency"] = _efficiency_block(rows_for_leg)
    return entry


def _paired_test(a_votes: dict, b_votes: dict) -> dict:
    """Pair only task_ids present in both vote maps, then hand off to `mcnemar`."""
    common = sorted(set(a_votes) & set(b_votes))
    result = mcnemar([a_votes[t] for t in common], [b_votes[t] for t in common])
    result["n_tasks_paired"] = len(common)
    return result


def _paired_block(legs_rows: dict) -> dict:
    """legs_rows: leg_key -> rows. LFM2.5's reasoning_on/off legs are identified by
    regime, not slug: exactly one leg carries each of "reasoning_on"/"reasoning_off"
    in a t127 sweep, and every other leg is regime "default" (an anchor)."""
    by_regime: dict = {}
    for rows in legs_rows.values():
        by_regime.setdefault(rows[0]["regime"], []).append(rows)

    on_legs = by_regime.get("reasoning_on", [])
    off_legs = by_regime.get("reasoning_off", [])
    anchor_legs = by_regime.get("default", [])

    anchors: dict = {}
    vs_off = None

    if on_legs:
        on_votes = _task_votes(on_legs[0])
        for anchor_rows in anchor_legs:
            anchors[anchor_rows[0]["slug"]] = _paired_test(on_votes, _task_votes(anchor_rows))
        if off_legs:
            vs_off = _paired_test(on_votes, _task_votes(off_legs[0]))

    return {"reasoning_on_vs_anchors": anchors, "reasoning_on_vs_reasoning_off": vs_off}


def _overall_block(legs: dict) -> dict:
    ranked = sorted(
        ({"leg": key, "success_rate_pct": v["success_rate_pct"], "n": v["n"]}
         for key, v in legs.items()),
        key=lambda d: d["success_rate_pct"], reverse=True,
    )
    return {
        "n_legs": len(legs),
        "n_rows": sum(v["n"] for v in legs.values()),
        "legs_ranked_by_success": ranked,
    }


def analyze(rows: list) -> dict:
    """Turn result rows into per-leg success/tier/efficiency stats, paired McNemar
    tests, and a cross-leg overall summary. See module docstring for the majority-vote
    and averaging conventions."""
    legs_rows = _group_by(rows, lambda r: _leg_key(r["slug"], r["regime"]))
    legs = {key: _leg_entry(leg_rows) for key, leg_rows in legs_rows.items()}
    return {
        "legs": legs,
        "paired": _paired_block(legs_rows),
        "overall": _overall_block(legs),
    }
