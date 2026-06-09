"""Model-agnostic native tool-calling loop. `client.chat(messages, tools)` returns an
assistant message dict (OpenAI shape: content + optional tool_calls). We execute any
tool_calls via dispatch, append results, and repeat until the model answers (no tool_calls)
or we hit max_steps. Records success-relevant telemetry."""
import json
from dataclasses import dataclass, field
from lib.agentic.native.dispatch import dispatch_tool


@dataclass
class RunResult:
    final_text: str = ""
    n_tool_calls: int = 0
    n_steps: int = 0
    n_tokens: int = 0
    stalled: bool = False
    bad_calls: int = 0
    messages: list = field(default_factory=list)


def run_agent(client, goal: str, tools: list, max_steps: int = 8, dispatch=None) -> RunResult:
    if dispatch is None:
        dispatch = dispatch_tool
    msgs = [{"role": "user", "content": goal}]
    r = RunResult()
    for _ in range(max_steps):
        r.n_steps += 1
        msg = client.chat(msgs, tools)
        r.n_tokens += int(msg.get("_tokens", 0))
        msgs.append({k: v for k, v in msg.items() if not k.startswith("_")})
        calls = msg.get("tool_calls")
        if not calls:
            r.final_text = msg.get("content") or ""
            r.messages = msgs
            return r
        for c in calls:
            r.n_tool_calls += 1
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                r.bad_calls += 1
            out = dispatch(fn.get("name", ""), args)
            if not out["ok"]:
                r.bad_calls += 1
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                         "content": json.dumps(out["result"] if out["ok"] else out["error"], default=str)})
    r.stalled = True
    r.messages = msgs
    return r
