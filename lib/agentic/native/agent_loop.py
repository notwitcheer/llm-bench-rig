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
    n_reasoning_tokens: int = 0
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
        # not all clients (e.g. test doubles) track reasoning tokens; default to 0
        # so existing callers stay backward-compatible.
        r.n_reasoning_tokens += int(getattr(client, "last_reasoning_tokens", 0) or 0)
        if msg.get("_error"):
            # server rejected the turn (unparseable tool-call JSON or context overflow).
            r.bad_calls += 1
            err = str(msg["_error"])
            if "context size" in err or "exceeds" in err:
                # context is full; adding more can't help — end and let the diff be extracted.
                r.stalled = True
                r.messages = msgs
                return r
            msgs.append({"role": "user", "content":
                         "Your previous response could not be processed (server error: " + err
                         + "). If it was a tool call, re-send it as a SINGLE valid JSON tool "
                         "call with all newlines and quotes properly escaped (or run a simpler "
                         "command). Then continue working toward the fix."})
            continue
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
