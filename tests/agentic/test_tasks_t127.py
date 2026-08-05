"""Frozen t127 30-task suite: shape/tier/sha invariants + representative check() behavior
(one task per tier, correct transcript -> True, wrong transcript -> False). Offline, no
model — exercises only the deterministic `check` callables against synthetic final_text."""
from collections import Counter
from pathlib import Path

from lib.agentic.native.tasks_t127 import (
    T127_TASKS, SUITE_SHA, SUITE_JSON_PATH, SUITE_SHA_PATH, check, compute_sha,
)


def test_exactly_thirty_tasks():
    assert len(T127_TASKS) == 30


def test_ids_unique():
    ids = [t["id"] for t in T127_TASKS]
    assert len(ids) == len(set(ids))


def test_ten_per_tier():
    c = Counter(t["tier"] for t in T127_TASKS)
    assert c == {"single_tool": 10, "multi_step": 10, "multi_turn_if": 10}


def test_every_task_shaped_like_the_native_schema():
    for t in T127_TASKS:
        assert t["id"] and t["goal"] and t["tier"]
        assert callable(t["check"])
        assert isinstance(t["opt_calls"], int) and t["opt_calls"] >= 1


def test_suite_sha_is_frozen():
    assert SUITE_SHA == compute_sha(T127_TASKS)


def test_suite_sha_changes_if_a_task_changes():
    mutated = [dict(T127_TASKS[0])]
    mutated[0]["goal"] = mutated[0]["goal"] + " (mutated)"
    mutated += T127_TASKS[1:]
    assert compute_sha(mutated) != SUITE_SHA


# --- one representative task per tier: check() true on a correct transcript,
#     false on a wrong one ------------------------------------------------------------

def test_single_tool_check_correct_and_wrong():
    t = next(x for x in T127_TASKS if x["id"] == "coding_sum_evens")
    assert t["tier"] == "single_tool"
    assert check(t, "the sum of the even numbers is 110") is True
    assert check(t, "the sum of the even numbers is 999") is False


def test_multi_step_check_correct_and_wrong():
    t = next(x for x in T127_TASKS if x["id"] == "chain_vram_x2")
    assert t["tier"] == "multi_step"
    assert check(t, "the RTX 5090 has 32GB, doubled that is 64") is True
    assert check(t, "the RTX 5090 has 32GB, doubled that is 999") is False


def test_multi_turn_if_check_correct_and_wrong():
    t = next(x for x in T127_TASKS if x["id"] == "mtif_vram_json")
    assert t["tier"] == "multi_turn_if"
    assert check(t, '{"vram_gb": 32, "doubled": 64}') is True
    assert check(t, '{"vram_gb": 32, "doubled": 999}') is False       # wrong value
    assert check(t, '{"vram_gb": 32}') is False                        # missing key
    assert check(t, 'the doubled vram is 64') is False                 # not JSON at all


def test_multi_turn_if_forbids_is_enforced():
    t = next(x for x in T127_TASKS if x["id"] == "mtif_config_forbid_json")
    assert check(t, '{"vram_gb": 32}') is True
    assert check(t, 'the model reports {"vram_gb": 32}') is False      # forbidden word present


def test_multi_turn_if_survives_a_think_tag_prefix():
    # check_compliance strips everything up to the last </think>; the JSON-value
    # validation in tasks_t127 must mirror that so reasoning models aren't penalized.
    t = next(x for x in T127_TASKS if x["id"] == "mtif_vram_json")
    text = '<think>32*2=64</think>{"vram_gb": 32, "doubled": 64}'
    assert check(t, text) is True


# --- frozen artifact on disk matches the in-memory suite ----------------------------

def test_frozen_json_and_sha_files_exist_and_match():
    assert SUITE_JSON_PATH.exists(), f"missing {SUITE_JSON_PATH}; run `python3 -m lib.agentic.native.tasks_t127`"
    assert SUITE_SHA_PATH.exists(), f"missing {SUITE_SHA_PATH}; run `python3 -m lib.agentic.native.tasks_t127`"
    import json
    on_disk = json.loads(SUITE_JSON_PATH.read_text())
    assert len(on_disk) == 30
    assert {t["id"] for t in on_disk} == {t["id"] for t in T127_TASKS}
    assert SUITE_SHA_PATH.read_text().strip() == SUITE_SHA
