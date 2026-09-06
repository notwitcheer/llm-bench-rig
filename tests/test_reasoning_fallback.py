"""LLMClient reasoning-content fallback is counted and surfaced per eval.

When a think-on model returns an empty `content` and the answer sits in
`reasoning_content`, the client scores the reasoning text. That is a legitimate
rescue, but a row whose score leaned on it should say so: the client flags the
last call and keeps a running count, and every eval writes the count next to
`parse_failures` in its result dict. Fake clients without the attribute must
still work (getattr default 0).
"""
import sys
import types

import httpx
from lib.evals.base import LLMClient


def _client_with_responses(payloads: list[dict]) -> LLMClient:
    it = iter(payloads)
    client = LLMClient("http://fake/v1", "m", think=False)
    client._client.close()
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=next(it)))
    )
    return client


def _msg(content, reasoning=None):
    m = {"content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return {"choices": [{"message": m}], "usage": {"completion_tokens": 3}}


def test_client_flags_and_counts_reasoning_fallback():
    with _client_with_responses([
        _msg("A"),                      # normal: content scored
        _msg("", "the answer is B"),    # fallback: reasoning scored
        _msg("   ", "C"),               # whitespace content is empty too
        _msg("D", "ignored reasoning"),  # content wins when present
    ]) as client:
        assert client.reasoning_fallback_count == 0
        assert client.last_scored_reasoning_fallback is False

        assert client.chat([{"role": "user", "content": "q"}]) == "A"
        assert client.last_scored_reasoning_fallback is False
        assert client.reasoning_fallback_count == 0

        assert client.chat([{"role": "user", "content": "q"}]) == "the answer is B"
        assert client.last_scored_reasoning_fallback is True
        assert client.reasoning_fallback_count == 1

        assert client.chat([{"role": "user", "content": "q"}]) == "C"
        assert client.last_scored_reasoning_fallback is True
        assert client.reasoning_fallback_count == 2

        assert client.chat([{"role": "user", "content": "q"}]) == "D"
        assert client.last_scored_reasoning_fallback is False  # reset per call
        assert client.reasoning_fallback_count == 2


def test_both_empty_is_not_a_fallback():
    with _client_with_responses([_msg("", "")]) as client:
        assert client.chat([{"role": "user", "content": "q"}]) == ""
        assert client.last_scored_reasoning_fallback is False
        assert client.reasoning_fallback_count == 0


# --- eval wiring with a fake client ---

class _FallbackClient:
    """Answers 'A' every time; every other answer is marked as a reasoning fallback."""

    def __init__(self):
        self.last_completion_tokens = None
        self.reasoning_fallback_count = 0
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        self.last_completion_tokens = 2
        if self.calls % 2 == 0:
            self.reasoning_fallback_count += 1
        return "A"


class _BareClient:
    """Fake without the counter attribute, like the stubs in older tests."""

    def __init__(self):
        self.last_completion_tokens = None

    def chat(self, messages, **kw):
        self.last_completion_tokens = 2
        return "A"


def _patch_datasets(monkeypatch, n_test=10):
    item = {"question": "q?", "choices": ["a", "b", "c", "d"], "answer": 0}

    def _fake_load_dataset(*args, **kwargs):
        return {"dev": [item] * 5, "train": [item] * 5, "test": [item] * n_test}

    fake = types.ModuleType("datasets")
    fake.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_mmlu_records_reasoning_fallback_count(monkeypatch, tmp_path):
    from lib.evals.mmlu import MMLUEval
    _patch_datasets(monkeypatch, n_test=10)
    ev = MMLUEval(client=_FallbackClient(), results_dir=tmp_path)
    result = ev.evaluate(subjects=["anatomy"])
    assert result["reasoning_fallback_count"] == 5
    assert result["parse_failures"] == 0
    assert result["per_subject"]["anatomy"]["reasoning_fallback_count"] == 5


def test_mmlu_bare_client_defaults_to_zero(monkeypatch, tmp_path):
    from lib.evals.mmlu import MMLUEval
    _patch_datasets(monkeypatch, n_test=4)
    result = MMLUEval(client=_BareClient(), results_dir=tmp_path).evaluate(subjects=["anatomy"])
    assert result["reasoning_fallback_count"] == 0


def test_mmlu_count_is_per_eval_not_lifetime(monkeypatch, tmp_path):
    """A client shared across evals must not leak earlier fallbacks into this one."""
    from lib.evals.mmlu import MMLUEval
    _patch_datasets(monkeypatch, n_test=4)
    client = _FallbackClient()
    client.reasoning_fallback_count = 40  # from an earlier task on the same client
    result = MMLUEval(client=client, results_dir=tmp_path).evaluate(subjects=["anatomy"])
    assert result["reasoning_fallback_count"] == 2
