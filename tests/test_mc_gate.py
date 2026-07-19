"""t103 / ADR-0004 — MC completion-length instrument gate.

Degenerate serving output can still parse to letters (2026-07-16 sglang GDN
incident: MMLU graded 2-4% from delirium). Pace is the mechanical tell:
think-off MC answers run 1-7 tokens; the poisoned run averaged ~850.
"""
import pytest

from lib.evals.base import CompletionLengthGate, InstrumentGateError


def test_trips_on_degenerate_pace():
    gate = CompletionLengthGate(threshold=50, armed=True)
    with pytest.raises(InstrumentGateError) as exc:
        for _ in range(10):
            gate.observe(850)  # the 2026-07-16 incident pace
    assert exc.value.mean == 850
    assert exc.value.threshold == 50
    assert exc.value.n_samples == 10


def test_no_trip_under_min_samples():
    gate = CompletionLengthGate(threshold=50, armed=True)
    for _ in range(9):
        gate.observe(850)  # 9 < min_samples: never raises


def test_healthy_answers_do_not_trip():
    gate = CompletionLengthGate(threshold=50, armed=True)
    for _ in range(500):
        gate.observe(4)  # think-off MC answers run 1-7 tokens


def test_single_outlier_does_not_trip():
    gate = CompletionLengthGate(threshold=50, armed=True)
    for _ in range(24):
        gate.observe(3)
    gate.observe(200)  # one rambling answer in a full window: mean ~11


def test_window_rolls_catches_midsuite_onset():
    gate = CompletionLengthGate(threshold=50, armed=True)
    for _ in range(1000):
        gate.observe(3)  # hours of healthy suite
    with pytest.raises(InstrumentGateError):
        for _ in range(25):
            gate.observe(850)  # server degenerates at subject 30


def test_disarmed_logs_but_never_raises():
    gate = CompletionLengthGate(threshold=50, armed=False)
    for _ in range(30):
        gate.observe(850)
    assert gate.mean == 850


def test_none_threshold_disables():
    gate = CompletionLengthGate(threshold=None, armed=True)
    for _ in range(30):
        gate.observe(850)
    assert gate.mean == 850


def test_mean_none_before_first_observe():
    assert CompletionLengthGate(threshold=50, armed=True).mean is None


# --- LLMClient usage capture ---

import httpx

from lib.evals.base import LLMClient


def _client_with_response(payload: dict) -> LLMClient:
    client = LLMClient("http://fake/v1", "m", think=False)
    client._client.close()
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    )
    return client


def test_chat_captures_usage_completion_tokens():
    payload = {
        "choices": [{"message": {"content": "A"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7},
    }
    with _client_with_response(payload) as client:
        assert client.last_completion_tokens is None
        assert client.chat([{"role": "user", "content": "q"}]) == "A"
        assert client.last_completion_tokens == 7


def test_chat_estimates_when_usage_missing():
    text = "x" * 400  # no usage block -> chars//4 + 1 estimate
    payload = {"choices": [{"message": {"content": text}}]}
    with _client_with_response(payload) as client:
        client.chat([{"role": "user", "content": "q"}])
        assert client.last_completion_tokens == 101


def test_estimate_includes_reasoning_content():
    payload = {"choices": [{"message": {"content": "", "reasoning_content": "y" * 200}}]}
    with _client_with_response(payload) as client:
        client.chat([{"role": "user", "content": "q"}])
        assert client.last_completion_tokens == 51
