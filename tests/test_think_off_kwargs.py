"""Tests for LLMClient think-OFF chat_template_kwargs across reasoning-model families.

Root-cause bug (North-Mini-Code-1.0, cohere2moe): the client sent only
`{"enable_thinking": False}` when think was off — the Gemma/Qwen key. North-Mini's
chat template gates reasoning on `reasoning` / `reasoning_effort`, so it ignored the
kwarg and emitted a full <|START_THINKING|> trace on every question (0.2-1 q/s + the
answer parser overrun -> "unparsed", contaminating scores).

The fix: think-OFF sends all known suppression keys at once. Each template reads the
key it knows and ignores the rest (unknown kwargs are harmless to minja/jinja), so one
payload covers Gemma (enable_thinking), Cohere/North (reasoning, reasoning_effort), etc.
Verified end-to-end on capsule: reasoning:False drops North-Mini from 106 -> 4
completion tokens with an empty reasoning_content.
"""

import httpx
from unittest.mock import MagicMock, patch

from lib.evals.base import LLMClient


def _mock_response(content: str = "A") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content,
                                                       "reasoning_content": None}}]}
    return resp


def _captured_payload(client: LLMClient) -> dict:
    """Call chat() against a mocked transport and return the POSTed JSON payload."""
    with patch.object(client._client, "post", return_value=_mock_response()) as mock_post:
        client.chat([{"role": "user", "content": "What is 2+2? A) 3 B) 4"}])
    return mock_post.call_args.kwargs["json"]


def test_think_off_sends_cohere_reasoning_keys():
    """think=False must carry North-Mini's reasoning toggle, not just enable_thinking."""
    payload = _captured_payload(LLMClient("http://x/v1", "m", think=False))
    ctk = payload.get("chat_template_kwargs", {})
    assert ctk.get("reasoning") is False, f"missing reasoning:False -> {ctk!r}"
    assert ctk.get("reasoning_effort") == "none", f"missing reasoning_effort:none -> {ctk!r}"


def test_think_off_keeps_enable_thinking_for_gemma_qwen():
    """The existing Gemma/Qwen key must remain so those board models stay think-OFF."""
    ctk = _captured_payload(LLMClient("http://x/v1", "m", think=False)).get("chat_template_kwargs", {})
    assert ctk.get("enable_thinking") is False, f"enable_thinking regressed -> {ctk!r}"


def test_think_on_sends_no_template_kwargs():
    """think=True (default) must not inject any suppression kwargs (keep reasoning ON)."""
    payload = _captured_payload(LLMClient("http://x/v1", "m", think=True))
    assert "chat_template_kwargs" not in payload, f"unexpected kwargs on think-ON -> {payload!r}"
