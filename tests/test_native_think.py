"""LlamaClient reasoning on/off toggle + reasoning-token capture (t127 task 1).
Mirrors lib/evals/base.py's LLMClient think toggle onto the native client. client.py
calls the module-level `httpx.post(...)` directly (no Client/session instance), so we
monkeypatch that exact symbol."""
import lib.agentic.native.client as client_mod
from lib.agentic.native.client import LlamaClient
from lib.agentic.native.agent_loop import run_agent


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "" if status_code == 200 else "error"

    def json(self):
        return self._payload


def _server_reply(content="ok", reasoning_content=None, usage=None):
    msg = {"role": "assistant", "content": content, "tool_calls": None}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    payload = {"choices": [{"message": msg}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_think_false_includes_chat_template_kwargs(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse(_server_reply())

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient(think=False)
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert captured["json"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning": False,
        "reasoning_effort": "none",
    }


def test_think_true_omits_chat_template_kwargs(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse(_server_reply())

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient()  # think defaults to True
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert "chat_template_kwargs" not in captured["json"]


def test_reasoning_tokens_captured_from_usage(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse(_server_reply(
            reasoning_content="thinking...",
            usage={"completion_tokens": 30, "reasoning_tokens": 18},
        ))

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient(think=True)
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert c.last_reasoning_tokens == 18


def test_reasoning_tokens_estimated_from_text_when_usage_lacks_field(monkeypatch):
    # llama-server's usage object has no reasoning_tokens field at all (only
    # prompt/completion/total/cached) -- must fall back to a chars/4 estimate over
    # reasoning_content instead of silently reading 0, or the think-on vs think-off
    # token-efficiency metric is always zeroed on real runs.
    reasoning_text = "thinking..."  # 11 chars -> 11 // 4 + 1 == 3

    def fake_post(url, json, timeout):
        return _FakeResponse(_server_reply(
            reasoning_content=reasoning_text,
            usage={"completion_tokens": 30},
        ))

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient(think=True)
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert c.last_reasoning_tokens == len(reasoning_text) // 4 + 1
    assert c.last_reasoning_tokens != 0


def test_reasoning_tokens_zero_from_usage_is_not_overridden_by_estimate(monkeypatch):
    # explicit presence check, not truthiness: a legitimately-present usage.reasoning_tokens
    # of 0 must be trusted as-is, not treated as "missing" and replaced by the text estimate.
    def fake_post(url, json, timeout):
        return _FakeResponse(_server_reply(
            reasoning_content="thinking...",
            usage={"completion_tokens": 30, "reasoning_tokens": 0},
        ))

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient(think=True)
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert c.last_reasoning_tokens == 0


def test_reasoning_tokens_default_zero_with_no_usage_at_all(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse(_server_reply())  # no reasoning_content, no usage

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LlamaClient()
    c.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert c.last_reasoning_tokens == 0


class _ReasoningScriptedClient:
    """Test double exposing last_reasoning_tokens like the real LlamaClient, so
    run_agent's accumulation across turns can be verified without hitting HTTP."""
    def __init__(self, turns, reasoning_per_turn):
        self.turns = list(turns)
        self.reasoning_per_turn = list(reasoning_per_turn)
        self.last_reasoning_tokens = 0

    def chat(self, messages, tools):
        self.last_reasoning_tokens = self.reasoning_per_turn.pop(0)
        return self.turns.pop(0)


def test_run_agent_accumulates_reasoning_tokens_across_turns():
    turns = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "calc", "arguments": '{"expr": "1+1"}'}}]},
        {"role": "assistant", "content": "done", "tool_calls": None},
    ]
    client = _ReasoningScriptedClient(turns, reasoning_per_turn=[10, 5])
    r = run_agent(client, goal="g", tools=[], max_steps=5)
    assert r.n_reasoning_tokens == 15
