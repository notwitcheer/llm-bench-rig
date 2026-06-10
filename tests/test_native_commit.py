"""Commit-the-fix axis: a per-run mutating tool (apply_fix) + state-based verification.
A model that DESCRIBES a fix without calling apply_fix leaves world_state empty -> fails,
mirroring SWE-bench empty patches. Mac-side, no GPU."""
from lib.agentic.native.commit import make_commit_dispatch


# local minimal ScriptedClient (avoids a cross-test-module import dependency)
class ScriptedClient:
    """Returns pre-baked assistant messages; ignores tools."""
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


def test_apply_fix_writes_world_state():
    dispatch, world_state = make_commit_dispatch(base=lambda n, a: {"ok": False, "result": None, "error": "base"})
    out = dispatch("apply_fix", {"target": "vram_budget", "value": "16"})
    assert out == {"ok": True, "result": "ok", "error": ""}
    assert world_state == {"vram_budget": "16"}


def test_non_apply_fix_delegates_to_base():
    seen = []
    def base(name, args):
        seen.append((name, args)); return {"ok": True, "result": "delegated", "error": ""}
    dispatch, world_state = make_commit_dispatch(base=base)
    out = dispatch("read_file", {"path": "/data/config.txt"})
    assert out["result"] == "delegated"
    assert seen == [("read_file", {"path": "/data/config.txt"})]
    assert world_state == {}


def test_apply_fix_coerces_values_to_str():
    dispatch, world_state = make_commit_dispatch(base=lambda n, a: {"ok": False, "result": None, "error": ""})
    dispatch("apply_fix", {"target": "x", "value": 64})
    assert world_state == {"x": "64"}


from lib.agentic.native.commit import check_commit


def test_check_commit_passes_on_matching_state():
    task = {"target": "vram_budget", "expect": "16"}
    assert check_commit(task, {"vram_budget": "16"}) is True


def test_check_commit_fails_on_empty_state():
    task = {"target": "vram_budget", "expect": "16"}
    assert check_commit(task, {}) is False  # described but never committed


def test_check_commit_fails_on_wrong_value():
    task = {"target": "vram_budget", "expect": "16"}
    assert check_commit(task, {"vram_budget": "32"}) is False


def test_check_commit_tolerates_whitespace_and_int():
    task = {"target": "x", "expect": 64}
    assert check_commit(task, {"x": " 64 "}) is True
