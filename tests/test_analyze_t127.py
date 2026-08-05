"""analyze_t127.analyze(rows), offline: success rate + Wilson CI (per leg and per
tier), majority-vote + paired McNemar, and token/tool efficiency aggregates. Rows are
SYNTHETIC (no model) but use REAL task_ids from tasks_t127.T127_TASKS so the tier map
and opt_calls (for tool_eff) are the genuine ones, not stubs.

Expected values are computed two ways, matching this repo's existing paired/wilson_ci
test style (see test_native_paired.py, phaseb's own tests):
  - success rates: hand-summed successes/n, then `wilson_ci(k, n)` called directly
    for the expected CI (not hardcoded numbers).
  - McNemar: majority-vote outcomes worked out by hand per task_id, then `mcnemar(a, b)`
    called directly on those hand-derived lists for the expected dict.
  - tool_eff: `metrics.score_run(...)` called directly per row for the expected mean.
Reusing the underlying primitives to derive the expected value (rather than re-deriving
their math) is intentional -- analyze_t127 must not reimplement wilson_ci/mcnemar/
score_run, so the test proves it *calls* them correctly on the right groupings, not
that it reimplements their arithmetic.

Two tasks (chain_vram_x2, coding_factorial5, mtif_vram_json) get DIFFERENT rep counts
within the same leg (3, 2, 3 for the reasoning_on leg) specifically to distinguish
"mean per task, averaged over reps, then averaged across tasks" (what analyze_t127
documents) from a flat mean over all rows -- the two are only equal when every task has
the same number of reps, which real t127 legs do but this fixture deliberately does not.
"""
import pytest

from lib.agentic.native.analyze_t127 import analyze
from lib.agentic.native.paired import mcnemar
from lib.agentic.native.tasks_t127 import T127_TASKS
from lib.agentic.native import metrics
from lib.agentic.phaseb import wilson_ci

_OPT_CALLS = {t["id"]: t["opt_calls"] for t in T127_TASKS}

# real tier assignments, asserted below rather than assumed
TIER = {t["id"]: t["tier"] for t in T127_TASKS}
assert TIER["coding_factorial5"] == "single_tool"
assert TIER["coding_sum_evens"] == "single_tool"
assert TIER["chain_vram_x2"] == "multi_step"
assert TIER["mtif_vram_json"] == "multi_turn_if"


def mk_rows(slug, regime, think, task_id, oks, gen_tokens, reasoning_tokens,
            tool_calls, bad_calls, stalled):
    """One row per rep, zipping the parallel per-rep lists. All lists must be the
    same length (that length is the task's rep count for THIS leg)."""
    n = len(oks)
    assert len({len(gen_tokens), len(reasoning_tokens), len(tool_calls), len(bad_calls),
                len(stalled), n}) == 1
    return [
        {"slug": slug, "regime": regime, "think": think, "task_id": task_id, "rep": i,
         "ok": oks[i], "tool_calls": tool_calls[i], "bad_calls": bad_calls[i],
         "gen_tokens": gen_tokens[i], "reasoning_tokens": reasoning_tokens[i],
         "stalled": stalled[i]}
        for i in range(n)
    ]


# --- reasoning_on leg (lfm25) -- 4 tasks, deliberately uneven rep counts -------------
ON_ROWS = (
    mk_rows("lfm25", "reasoning_on", True, "coding_factorial5",
            oks=[True, True, False], gen_tokens=[100, 120, 110], reasoning_tokens=[50, 60, 40],
            tool_calls=[1, 1, 2], bad_calls=[0, 0, 1], stalled=[False, False, False])
    + mk_rows("lfm25", "reasoning_on", True, "chain_vram_x2",
              oks=[True, False], gen_tokens=[200, 220], reasoning_tokens=[80, 100],
              tool_calls=[2, 3], bad_calls=[0, 1], stalled=[False, True])
    + mk_rows("lfm25", "reasoning_on", True, "mtif_vram_json",
              oks=[True, True, True], gen_tokens=[150, 150, 150], reasoning_tokens=[70, 70, 70],
              tool_calls=[1, 1, 1], bad_calls=[0, 0, 0], stalled=[False, False, False])
    + mk_rows("lfm25", "reasoning_on", True, "coding_sum_evens",
              oks=[True, True], gen_tokens=[90, 90], reasoning_tokens=[30, 30],
              tool_calls=[1, 1], bad_calls=[0, 0], stalled=[False, False])
)

# --- reasoning_off leg (same slug, same 4 tasks, different outcomes) -----------------
OFF_ROWS = (
    mk_rows("lfm25", "reasoning_off", False, "coding_factorial5",
            oks=[False, False, True], gen_tokens=[80, 80, 80], reasoning_tokens=[0, 0, 0],
            tool_calls=[1, 1, 1], bad_calls=[0, 0, 0], stalled=[False, False, False])
    + mk_rows("lfm25", "reasoning_off", False, "chain_vram_x2",
              oks=[True, True], gen_tokens=[150, 150], reasoning_tokens=[0, 0],
              tool_calls=[2, 2], bad_calls=[0, 0], stalled=[False, False])
    + mk_rows("lfm25", "reasoning_off", False, "mtif_vram_json",
              oks=[True, False, False], gen_tokens=[100, 100, 100], reasoning_tokens=[0, 0, 0],
              tool_calls=[1, 1, 1], bad_calls=[0, 0, 0], stalled=[False, False, False])
    + mk_rows("lfm25", "reasoning_off", False, "coding_sum_evens",
              oks=[False, False], gen_tokens=[70, 70], reasoning_tokens=[0, 0],
              tool_calls=[1, 1], bad_calls=[0, 0], stalled=[False, False])
)

# --- anchor A (default regime) -- only 3 of the 4 tasks, to exercise the "pair only
# task_ids present in BOTH legs" rule (coding_sum_evens absent here) ------------------
ANCHOR_A_ROWS = (
    mk_rows("anchorA", "default", True, "coding_factorial5",
            oks=[True, True], gen_tokens=[100, 100], reasoning_tokens=[0, 0],
            tool_calls=[1, 1], bad_calls=[0, 0], stalled=[False, False])
    + mk_rows("anchorA", "default", True, "chain_vram_x2",
              oks=[False, False], gen_tokens=[200, 200], reasoning_tokens=[0, 0],
              tool_calls=[2, 2], bad_calls=[0, 0], stalled=[False, False])
    + mk_rows("anchorA", "default", True, "mtif_vram_json",
              oks=[True, False], gen_tokens=[150, 150], reasoning_tokens=[0, 0],
              tool_calls=[1, 1], bad_calls=[0, 0], stalled=[False, False])
)

# --- anchor B (default regime) -- all 4 tasks, single rep each ----------------------
ANCHOR_B_ROWS = (
    mk_rows("anchorB", "default", True, "coding_factorial5",
            oks=[False], gen_tokens=[100], reasoning_tokens=[0],
            tool_calls=[1], bad_calls=[0], stalled=[False])
    + mk_rows("anchorB", "default", True, "chain_vram_x2",
              oks=[False], gen_tokens=[200], reasoning_tokens=[0],
              tool_calls=[2], bad_calls=[0], stalled=[False])
    + mk_rows("anchorB", "default", True, "mtif_vram_json",
              oks=[False], gen_tokens=[150], reasoning_tokens=[0],
              tool_calls=[1], bad_calls=[0], stalled=[False])
    + mk_rows("anchorB", "default", True, "coding_sum_evens",
              oks=[True], gen_tokens=[90], reasoning_tokens=[0],
              tool_calls=[1], bad_calls=[0], stalled=[False])
)

ALL_ROWS = list(ON_ROWS) + list(OFF_ROWS) + list(ANCHOR_A_ROWS) + list(ANCHOR_B_ROWS)


# ---------------------------------------------------------------------------
# success rate + Wilson CI, per leg and per tier
# ---------------------------------------------------------------------------

def test_leg_success_rate_and_ci_reasoning_on():
    result = analyze(ALL_ROWS)
    leg = result["legs"]["lfm25|reasoning_on"]
    # 10 rows total (3+2+3+2), 8 successes (2+1+3+2)
    assert leg["n"] == 10
    assert leg["successes"] == 8
    assert leg["success_rate_pct"] == 80.0
    assert leg["success_ci_pct"] == wilson_ci(8, 10)
    assert leg["think"] is True
    assert leg["slug"] == "lfm25"
    assert leg["regime"] == "reasoning_on"


def test_leg_success_rate_and_ci_reasoning_off():
    leg = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_off"]
    # 10 rows, successes: 1 (coding_factorial5) + 2 (chain_vram_x2) + 1 (mtif_vram_json) + 0
    assert leg["n"] == 10
    assert leg["successes"] == 4
    assert leg["success_rate_pct"] == 40.0
    assert leg["success_ci_pct"] == wilson_ci(4, 10)


def test_per_tier_success_reasoning_on():
    tiers = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_on"]["tiers"]
    # single_tool = coding_factorial5 (n=3,k=2) + coding_sum_evens (n=2,k=2) -> n=5,k=4
    st = tiers["single_tool"]
    assert st["n"] == 5
    assert st["successes"] == 4
    assert st["success_rate_pct"] == 80.0
    assert st["success_ci_pct"] == wilson_ci(4, 5)
    # multi_step = chain_vram_x2 only -> n=2, k=1
    ms = tiers["multi_step"]
    assert ms["n"] == 2
    assert ms["successes"] == 1
    assert ms["success_ci_pct"] == wilson_ci(1, 2)
    # multi_turn_if = mtif_vram_json only -> n=3, k=3
    mt = tiers["multi_turn_if"]
    assert mt["n"] == 3
    assert mt["successes"] == 3
    assert mt["success_ci_pct"] == wilson_ci(3, 3)


def test_per_tier_success_reasoning_off_multi_step():
    tiers = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_off"]["tiers"]
    ms = tiers["multi_step"]
    assert ms["n"] == 2
    assert ms["successes"] == 2
    assert ms["success_ci_pct"] == wilson_ci(2, 2)


# ---------------------------------------------------------------------------
# majority vote + paired McNemar
# ---------------------------------------------------------------------------

def test_reasoning_on_vs_reasoning_off_mcnemar():
    # Majority vote per task_id (ties -> False), worked out by hand from the fixture:
    on_votes = {"coding_factorial5": True,   # 2/3 True
                "chain_vram_x2": False,      # 1/2 -> tie -> False
                "mtif_vram_json": True,      # 3/3 True
                "coding_sum_evens": True}    # 2/2 True
    off_votes = {"coding_factorial5": False,  # 1/3 True -> majority False
                 "chain_vram_x2": True,       # 2/2 True
                 "mtif_vram_json": False,     # 1/3 True -> majority False
                 "coding_sum_evens": False}   # 0/2 True
    common = sorted(on_votes)
    expected = mcnemar([on_votes[t] for t in common], [off_votes[t] for t in common])

    got = analyze(ALL_ROWS)["paired"]["reasoning_on_vs_reasoning_off"]
    assert got["n_tasks_paired"] == 4
    assert got["b01"] == expected["b01"] == 1   # chain_vram_x2: on=False, off=True
    assert got["b10"] == expected["b10"] == 3   # the other 3: on=True, off=False
    assert got["n_discordant"] == expected["n_discordant"] == 4
    assert got["p"] == pytest.approx(expected["p"], abs=1e-9)
    assert got["p"] == pytest.approx(0.625, abs=1e-9)


def test_reasoning_on_vs_anchor_a_pairs_only_shared_task_ids():
    # anchorA has no coding_sum_evens row at all -> must be excluded from pairing.
    on_votes = {"coding_factorial5": True, "chain_vram_x2": False, "mtif_vram_json": True}
    a_votes = {"coding_factorial5": True, "chain_vram_x2": False, "mtif_vram_json": False}
    common = sorted(on_votes)
    expected = mcnemar([on_votes[t] for t in common], [a_votes[t] for t in common])

    got = analyze(ALL_ROWS)["paired"]["reasoning_on_vs_anchors"]["anchorA"]
    assert got["n_tasks_paired"] == 3   # NOT 4 -- coding_sum_evens excluded
    assert got["b01"] == expected["b01"] == 0
    assert got["b10"] == expected["b10"] == 1
    assert got["p"] == pytest.approx(expected["p"], abs=1e-9)


def test_reasoning_on_vs_anchor_b_all_four_tasks_paired():
    on_votes = {"coding_factorial5": True, "chain_vram_x2": False,
                "mtif_vram_json": True, "coding_sum_evens": True}
    b_votes = {"coding_factorial5": False, "chain_vram_x2": False,
               "mtif_vram_json": False, "coding_sum_evens": True}
    common = sorted(on_votes)
    expected = mcnemar([on_votes[t] for t in common], [b_votes[t] for t in common])

    got = analyze(ALL_ROWS)["paired"]["reasoning_on_vs_anchors"]["anchorB"]
    assert got["n_tasks_paired"] == 4
    assert got["b01"] == expected["b01"] == 0
    assert got["b10"] == expected["b10"] == 2   # coding_factorial5, mtif_vram_json
    assert got["p"] == pytest.approx(expected["p"], abs=1e-9)
    assert got["p"] == pytest.approx(0.5, abs=1e-9)


def test_paired_has_an_entry_per_anchor():
    anchors = analyze(ALL_ROWS)["paired"]["reasoning_on_vs_anchors"]
    assert set(anchors) == {"anchorA", "anchorB"}


# ---------------------------------------------------------------------------
# efficiency aggregates
# ---------------------------------------------------------------------------

def test_efficiency_reasoning_on_leg():
    eff = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_on"]["efficiency"]
    # per-task means (averaged over that task's reps FIRST -- the 2-rep chain_vram_x2
    # task must not get double weight vs the 3-rep tasks):
    #   coding_factorial5: gen=(100+120+110)/3=110, reason=(50+60+40)/3=50
    #   chain_vram_x2:     gen=(200+220)/2=210,      reason=(80+100)/2=90
    #   mtif_vram_json:    gen=150,                  reason=70
    #   coding_sum_evens:  gen=90,                   reason=30
    # mean over the 4 tasks:
    assert eff["mean_gen_tokens_per_task"] == pytest.approx((110 + 210 + 150 + 90) / 4)
    assert eff["mean_reasoning_tokens_per_task"] == pytest.approx((50 + 90 + 70 + 30) / 4)
    assert eff["mean_total_tokens_per_task"] == pytest.approx(
        eff["mean_gen_tokens_per_task"] + eff["mean_reasoning_tokens_per_task"])

    # success-per-1k-tokens uses the RAW total over all 10 rows (not the per-task
    # mean) -- total successes (8) / (total tokens summed over every row / 1000).
    total_tokens = (
        (100 + 50) + (120 + 60) + (110 + 40)      # coding_factorial5, 3 rows
        + (200 + 80) + (220 + 100)                 # chain_vram_x2, 2 rows
        + (150 + 70) * 3                            # mtif_vram_json, 3 rows
        + (90 + 30) * 2                              # coding_sum_evens, 2 rows
    )
    assert total_tokens == 1980
    assert eff["success_per_1k_tokens"] == pytest.approx(8 / (1980 / 1000))

    # mean_tool_calls / mean_bad_calls, same per-task-then-across-tasks averaging.
    # analyze_t127 rounds these to 2dp for display (repo style, see metrics.py), so
    # compare with an absolute tolerance wide enough to absorb that rounding.
    assert eff["mean_tool_calls"] == pytest.approx(
        (((1 + 1 + 2) / 3) + ((2 + 3) / 2) + 1.0 + 1.0) / 4, abs=0.005)
    assert eff["mean_bad_calls"] == pytest.approx(
        (((0 + 0 + 1) / 3) + ((0 + 1) / 2) + 0.0 + 0.0) / 4, abs=0.005)


def test_efficiency_tool_eff_reuses_score_run():
    on_rows = [r for r in ALL_ROWS if r["slug"] == "lfm25" and r["regime"] == "reasoning_on"]
    expected_scores = [
        metrics.score_run(r["ok"], r["tool_calls"], _OPT_CALLS[r["task_id"]],
                           r["bad_calls"], r["stalled"])["tool_eff"]
        for r in on_rows
    ]
    expected_mean = sum(expected_scores) / len(expected_scores)

    got = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_on"]["efficiency"]["mean_tool_eff"]
    assert got == pytest.approx(expected_mean, abs=1e-3)
    # sanity: not a degenerate 1.0/0.0 -- proves bad_calls/extra-calls actually bite
    assert 0.0 < expected_mean < 1.0


def test_efficiency_reasoning_off_has_zero_reasoning_tokens():
    eff = analyze(ALL_ROWS)["legs"]["lfm25|reasoning_off"]["efficiency"]
    assert eff["mean_reasoning_tokens_per_task"] == 0.0


# ---------------------------------------------------------------------------
# overall
# ---------------------------------------------------------------------------

def test_overall_counts_and_ranking():
    overall = analyze(ALL_ROWS)["overall"]
    assert overall["n_legs"] == 4
    assert overall["n_rows"] == len(ALL_ROWS) == 10 + 10 + 6 + 4
    ranked = overall["legs_ranked_by_success"]
    assert [r["leg"] for r in ranked] == [
        "lfm25|reasoning_on", "anchorA|default", "lfm25|reasoning_off", "anchorB|default",
    ]
    assert [r["success_rate_pct"] for r in ranked] == [80.0, 50.0, 40.0, 25.0]


def test_analyze_top_level_shape():
    result = analyze(ALL_ROWS)
    assert set(result) == {"legs", "paired", "overall"}
    assert set(result["legs"]) == {
        "lfm25|reasoning_on", "lfm25|reasoning_off", "anchorA|default", "anchorB|default",
    }
