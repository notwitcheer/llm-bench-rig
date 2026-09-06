"""Served (http) speed lane: sse parsing, summariser and lane loop, no network."""
import hashlib
import json

import pytest

from lib import speed_served as ss
from lib.workloads import WORKLOADS, WORKLOAD_NAMES


# --- canned sse stream ---

def _chunk(content=None, finish=None, usage=None, timings=None):
    d = {"choices": [{"delta": {"content": content} if content is not None else {},
                      "finish_reason": finish}]}
    if usage is not None:
        d["usage"] = usage
    if timings is not None:
        d["timings"] = timings
    return "data: " + json.dumps(d)


TIMINGS = {"prompt_n": 12, "prompt_ms": 40.0, "predicted_n": 5, "predicted_ms": 100.0,
           "predicted_per_second": 50.0, "draft_n": 8, "draft_n_accepted": 6}

CANNED = [
    ": keep-alive comment, ignored",
    "",
    _chunk(content=None),                 # role-only first delta, not content
    _chunk(content="Hello"),              # first content chunk -> ttft
    "data: {not json",                    # tolerated
    _chunk(content=", world"),
    _chunk(content="!", finish="stop"),
    "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                           "timings": TIMINGS}),
    "data: [DONE]",
    _chunk(content="never read"),         # after DONE
]


class FakeClock:
    def __init__(self, ticks):
        self.ticks = iter(ticks)

    def __call__(self):
        return next(self.ticks)


def test_parse_sse_stream_canned_chunks():
    # clock calls: first content chunk, then end
    clock = FakeClock([100.25, 101.0])
    rec = ss.parse_sse_stream(CANNED, clock=clock, t0=100.0)
    assert rec["ttft_s"] == 0.25
    assert rec["total_s"] == 1.0
    assert rec["completion_tokens"] == 5
    assert rec["prompt_tokens"] == 12
    assert rec["timings"] == TIMINGS
    assert rec["finish_reason"] == "stop"
    assert rec["sha256"] == hashlib.sha256(b"Hello, world!").hexdigest()
    assert rec["text_head"] == "Hello, world!"
    assert rec["has_think_tag"] is False


def test_parse_sse_stream_no_content_gives_none_ttft():
    lines = [_chunk(content=None, finish="length"),
             "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 0, "prompt_tokens": 3}}),
             "data: [DONE]"]
    rec = ss.parse_sse_stream(lines, clock=FakeClock([5.0]), t0=4.0)
    assert rec["ttft_s"] is None
    assert rec["total_s"] == 1.0
    assert rec["completion_tokens"] == 0
    assert rec["sha256"] == hashlib.sha256(b"").hexdigest()


def test_parse_sse_stream_accepts_bytes_style_newlines():
    lines = [l + "\n" for l in CANNED]
    rec = ss.parse_sse_stream(lines, clock=FakeClock([1.5, 2.0]), t0=1.0)
    assert rec["text_head"] == "Hello, world!"
    assert rec["ttft_s"] == 0.5


# --- request body / url ---

def test_build_request_streams_with_usage_and_cache_flag():
    body = ss.build_request([{"role": "user", "content": "x"}], 64, cache_prompt=False)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["cache_prompt"] is False
    assert body["temperature"] == 0
    assert body["max_tokens"] == 64
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("api", ["http://h:8090", "http://h:8090/", "http://h:8090/v1", "http://h:8090/v1/"])
def test_chat_completions_url(api):
    assert ss.chat_completions_url(api) == "http://h:8090/v1/chat/completions"


# --- summariser ---

def _rec(ttft, total, tokens, pred=None, draft=None, sha="a", workload="prose"):
    t = {}
    if pred is not None:
        t["predicted_per_second"] = pred
    if draft is not None:
        t["draft_n"], t["draft_n_accepted"] = draft
    return {"ttft_s": ttft, "total_s": total, "completion_tokens": tokens,
            "timings": t or None, "sha256": sha, "workload": workload}


def test_percentile_interpolates_like_numpy():
    assert ss.percentile([1, 2, 3, 4], 50) == 2.5
    assert ss.percentile([4, 1, 3, 2], 90) == pytest.approx(3.7)
    assert ss.percentile([7], 90) == 7.0
    assert ss.percentile([], 50) is None
    assert ss.percentile([None, 2.0], 50) == 2.0


def test_perceived_and_total_tps():
    r = _rec(0.5, 2.5, 100)
    assert ss.perceived_tps(r) == pytest.approx(50.0)   # 100 / (2.5 - 0.5)
    assert ss.total_tps(r) == pytest.approx(40.0)       # 100 / 2.5
    assert ss.perceived_tps(_rec(None, 2.5, 100)) is None
    assert ss.perceived_tps(_rec(2.5, 2.5, 100)) is None  # zero decode window
    assert ss.perceived_tps(_rec(0.5, 2.5, 0)) is None


def test_summarise_percentiles_and_acceptance():
    recs = [
        _rec(0.10, 2.10, 200, pred=100.0, draft=(10, 8)),   # perceived 100
        _rec(0.20, 1.20, 200, pred=200.0, draft=(10, 6)),   # perceived 200
        _rec(0.30, 0.80, 200, pred=400.0, draft=(20, 10)),  # perceived 400
        _rec(0.40, 4.40, 200, pred=50.0, sha="b"),          # perceived 50, no draft
    ]
    s = ss.summarise(recs)
    assert s["n"] == 4
    assert s["n_no_content"] == 0
    assert s["ttft_p50_s"] == pytest.approx(0.25)
    assert s["ttft_p90_s"] == pytest.approx(0.37)
    assert s["perceived_tps_p50"] == pytest.approx(150.0)
    assert s["perceived_tps_p90"] == pytest.approx(340.0)
    assert s["server_predicted_tps_p50"] == pytest.approx(150.0)
    assert s["acceptance_rate"] == pytest.approx(24 / 40)
    assert s["draft_n"] == 40 and s["draft_n_accepted"] == 24
    assert s["distinct_outputs"] == 2


def test_summarise_without_draft_reports_none_and_counts_no_content():
    recs = [_rec(0.1, 1.1, 10), _rec(None, 1.0, 0)]
    s = ss.summarise(recs)
    assert s["acceptance_rate"] is None
    assert s["draft_n"] is None
    assert s["server_predicted_tps_p50"] is None
    assert s["n_no_content"] == 1
    assert s["perceived_tps_p50"] == pytest.approx(10.0)


def test_summarise_group_by_workload():
    recs = [_rec(0.1, 1.1, 10, workload="code"), _rec(0.3, 1.3, 10, workload="chat"),
            _rec(0.5, 1.5, 10, workload="code")]
    g = ss.summarise(recs, group_by="workload")
    assert set(g) == {"code", "chat"}
    assert g["code"]["n"] == 2 and g["code"]["ttft_p50_s"] == pytest.approx(0.3)
    assert g["chat"]["n"] == 1


# --- lane loop with a fake client ---

def test_run_served_lane_warmup_discarded_and_records_stamped(tmp_path):
    calls = []

    def fake_chat(api, messages, max_tokens, cache_prompt=True):
        calls.append((messages[0]["content"], max_tokens, cache_prompt))
        return {"ttft_s": 0.1, "total_s": 1.1, "completion_tokens": 10, "prompt_tokens": 5,
                "timings": None, "sha256": "x", "text_head": "", "has_think_tag": False}

    wl = {"prose": ["p1", "p2"], "chat": ["c1"]}
    out = tmp_path / "lane.jsonl"
    recs = ss.run_served_lane("http://h", "mtp_n2", out, repeats=2, max_tokens=32,
                              cache_prompt=False, workloads=wl, log=lambda *_: None,
                              chat=fake_chat)
    # warm-up first, not recorded
    assert calls[0] == (ss.WARMUP_PROMPT, ss.WARMUP_TOKENS, False)
    assert len(calls) == 1 + 2 * 3
    assert len(recs) == 6
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 6
    assert [(r["workload"], r["idx"], r["rep"]) for r in lines] == [
        ("prose", 0, 0), ("prose", 1, 0), ("chat", 0, 0),
        ("prose", 0, 1), ("prose", 1, 1), ("chat", 0, 1)]
    for r in lines:
        assert r["mode"] == "mtp_n2"
        assert r["cache_prompt"] is False
        assert r["max_tokens"] == 32
        assert "ts" in r
    # appending twice keeps the earlier records
    ss.run_served_lane("http://h", "base", out, workloads={"chat": ["c1"]},
                       log=lambda *_: None, chat=fake_chat)
    assert len(ss.load_jsonl(out)) == 7


# --- workload set ---

def test_workloads_four_by_eight():
    assert WORKLOAD_NAMES == ("prose", "code", "repetitive", "chat")
    assert all(len(v) == 8 for v in WORKLOADS.values())
    assert all(isinstance(p, str) and p for v in WORKLOADS.values() for p in v)
    assert len({p for v in WORKLOADS.values() for p in v}) == 32  # no duplicates


def test_iter_workload_requests_order():
    seq = list(ss.iter_workload_requests({"a": ["x", "y"], "b": ["z"]}, repeats=2))
    assert seq == [(0, "a", 0, "x"), (0, "a", 1, "y"), (0, "b", 0, "z"),
                   (1, "a", 0, "x"), (1, "a", 1, "y"), (1, "b", 0, "z")]
