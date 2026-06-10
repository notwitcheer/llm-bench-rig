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


from lib.agentic.native.agent_loop import run_agent


def test_committer_passes_describer_fails_end_to_end():
    """The keystone: committing vs describing, through the real run_agent loop."""
    task = {"id": "commit_demo", "axis": "commit", "opt_calls": 1, "target": "vram_budget", "expect": "16"}

    # committer: reads, then CALLS apply_fix with the right value, then answers
    committer_turns = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/data/config.txt"}'}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function",
            "function": {"name": "apply_fix", "arguments": '{"target": "vram_budget", "value": "16"}'}}]},
        {"role": "assistant", "content": "done, budget set to 16", "tool_calls": None},
    ]
    dispatch, world_state = make_commit_dispatch()
    run_agent(ScriptedClient(committer_turns), "g", tools=[], max_steps=8, dispatch=dispatch)
    assert check_commit(task, world_state) is True

    # describer: reads, then only DESCRIBES the fix in prose (never calls apply_fix)
    describer_turns = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/data/config.txt"}'}}]},
        {"role": "assistant", "content": "the vram budget should be 16", "tool_calls": None},
    ]
    dispatch2, world_state2 = make_commit_dispatch()
    run_agent(ScriptedClient(describer_turns), "g", tools=[], max_steps=8, dispatch=dispatch2)
    assert check_commit(task, world_state2) is False
    assert world_state2 == {}


from lib.agentic.native.tools_ext import COMMIT_TOOLS
from lib.agentic.native.schemas import to_openai_tools
from lib.agentic.native.tasks import TASKS


def test_commit_tools_schema_parses():
    schemas = to_openai_tools(COMMIT_TOOLS)
    fn = schemas[0]["function"]
    assert fn["name"] == "apply_fix"
    assert set(fn["parameters"]["properties"]) == {"target", "value"}


def test_commit_tasks_well_formed():
    commit = [t for t in TASKS if t["axis"] == "commit"]
    assert len(commit) >= 6
    for t in commit:
        assert t["target"] and t["expect"] and t["goal"] and "opt_calls" in t
        assert "apply_fix" in t["goal"]  # the task names the commit tool
