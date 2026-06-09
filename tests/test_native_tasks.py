from lib.agentic.native.tasks import TASKS, check


def test_tasks_present_and_shaped():
    assert len(TASKS) >= 40
    AXES = {"chain", "multistep", "coding", "error_recovery", "distractor", "long_context"}
    for t in TASKS:
        assert t["id"] and t["goal"] and t["axis"] in AXES
        assert callable(t["check"]) or t.get("answer") is not None
        assert isinstance(t["opt_calls"], int) and t["opt_calls"] >= 1


def test_short_axes_have_enough_tasks():
    from collections import Counter
    c = Counter(t["axis"] for t in TASKS)
    for ax in ("chain", "multistep", "coding"):
        assert c[ax] >= 8, f"{ax} has {c[ax]}"


def test_calc_chain_checker_accepts_right_answer():
    t = next(t for t in TASKS if t["id"] == "chain_vram_x2")
    # goal: search rtx 5090 vram (32) then calc 32*2 -> 64
    assert check(t, final_text="the answer is 64") is True
    assert check(t, final_text="it is 128") is False


def test_numeric_checker_is_word_boundary():
    # answer "16" must not be satisfied by "1600" but must survive a trailing period
    t = next(t for t in TASKS if t["id"] == "chain_config_half")
    assert check(t, final_text="result: 16") is True
    assert check(t, final_text="result: 1600") is False
    assert check(t, final_text="The half is 16.") is True      # sentence-final period
    assert check(t, final_text="around 16, roughly") is True   # comma after
    assert check(t, final_text="16.5 GB") is False             # real decimal -> not 16


def test_ids_unique():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids))


def test_recovery_and_distractor_present():
    from collections import Counter
    c = Counter(t["axis"] for t in TASKS)
    assert c["error_recovery"] >= 5 and c["distractor"] >= 5


def test_distractor_lure_is_not_the_answer():
    # the read_cache lure says 24GB; the right VRAM answer is 32 — lure must fail the check
    t = next(t for t in TASKS if t["id"] == "distractor_vram_lure")
    assert check(t, "the cache said 24") is False
    assert check(t, "the vram is 32") is True


def test_long_context_tasks_shaped():
    lc = [t for t in TASKS if t["axis"] == "long_context"]
    assert len(lc) >= 4
    for t in lc:
        assert t["ctx_tier"] in {"32k", "128k"}
    assert {t["ctx_tier"] for t in lc} == {"32k", "128k"}      # both tiers present


def test_long_context_answer_matches_needle():
    from lib.agentic.native.tools_ext import vault_code
    t = next(t for t in TASKS if t["id"] == "longctx_32k_late")
    assert check(t, f"the code is {vault_code(102)}") is True
