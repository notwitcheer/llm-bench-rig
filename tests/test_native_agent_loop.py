from lib.agentic.native.agent_loop import run_agent, RunResult


class ScriptedClient:
    """Returns pre-baked assistant messages; records the tool results it was fed."""
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_tool_msgs = []

    def chat(self, messages, tools):
        for m in messages:
            if m.get("role") == "tool":
                self.seen_tool_msgs.append(m["content"])
        return self.turns.pop(0)


def test_loop_executes_tool_then_finishes():
    turns = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "calc", "arguments": '{"expr": "21*2"}'}}]},
        {"role": "assistant", "content": "the answer is 42", "tool_calls": None},
    ]
    r = run_agent(ScriptedClient(turns), goal="double 21", tools=[], max_steps=5)
    assert isinstance(r, RunResult)
    assert r.final_text == "the answer is 42"
    assert r.n_tool_calls == 1
    assert r.stalled is False


def test_loop_stops_at_step_cap():
    loop_turn = {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c", "type": "function",
                                 "function": {"name": "calc", "arguments": '{"expr":"1+1"}'}}]}
    r = run_agent(ScriptedClient([loop_turn] * 10), goal="g", tools=[], max_steps=3)
    assert r.stalled is True and r.n_steps == 3


def test_malformed_arguments_counted_as_bad_call():
    turns = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "calc", "arguments": "{not json"}}]},
        {"role": "assistant", "content": "done", "tool_calls": None},
    ]
    r = run_agent(ScriptedClient(turns), goal="g", tools=[], max_steps=5)
    assert r.bad_calls >= 1


def test_run_agent_uses_injected_dispatch():
    seen = []
    def my_dispatch(name, args):
        seen.append((name, args))
        return {"ok": True, "result": "tool-result", "error": ""}
    turns = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "do_thing", "arguments": '{"x": 1}'}}]},
        {"role": "assistant", "content": "done", "tool_calls": None},
    ]
    r = run_agent(ScriptedClient(turns), goal="g", tools=[], max_steps=5, dispatch=my_dispatch)
    assert seen == [("do_thing", {"x": 1})]
    assert r.final_text == "done"
