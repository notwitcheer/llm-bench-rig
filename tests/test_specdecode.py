from lib.specdecode import (
    acceptance_length,
    acceptance_rate,
    delta,
    parse_spec_metrics,
    speedup,
    summarize_acceptance,
)

# A trimmed vLLM /metrics dump (v1 names, with labels + _total suffix + unrelated lines).
SAMPLE = """\
# HELP vllm:num_requests_running Number of requests running.
vllm:num_requests_running{model_name="google/gemma-4-26B-A4B-it"} 1.0
vllm:spec_decode_num_drafts_total{model_name="google/gemma-4-26B-A4B-it"} 1000.0
vllm:spec_decode_num_draft_tokens_total{model_name="google/gemma-4-26B-A4B-it"} 15000.0
vllm:spec_decode_num_accepted_tokens_total{model_name="google/gemma-4-26B-A4B-it"} 8000.0
vllm:gpu_cache_usage_perc{model_name="google/gemma-4-26B-A4B-it"} 0.42
"""


def test_parse_pulls_spec_family_ignores_rest():
    m = parse_spec_metrics(SAMPLE)
    assert m["num_drafts"] == 1000.0
    assert m["num_draft_tokens"] == 15000.0
    assert m["num_accepted_tokens"] == 8000.0
    # non-spec lines must not leak in
    assert "num_requests_running" not in m
    assert all(not k.startswith("gpu_cache") for k in m)


def test_parse_tolerates_no_total_suffix_and_no_labels():
    txt = "vllm:spec_decode_num_drafts 5\nvllm:spec_decode_num_accepted_tokens 40\n"
    m = parse_spec_metrics(txt)
    assert m["num_drafts"] == 5.0 and m["num_accepted_tokens"] == 40.0


def test_parse_ignores_per_pos_histogram_and_created_decoys():
    # The real vLLM 0.21 dump: the _per_pos histogram (0.0) and _created timestamp sit right
    # next to the real total and collide by prefix. The real total (189) must win, never 0.0.
    txt = (
        "vllm:spec_decode_num_accepted_tokens_total{model_name=\"m\"} 189.0\n"
        "vllm:spec_decode_num_accepted_tokens_created{model_name=\"m\"} 1.781341706e+09\n"
        "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 0.0\n"
        "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"} 0.0\n"
        "vllm:spec_decode_num_drafts_total{model_name=\"m\"} 161.0\n"
        "vllm:spec_decode_num_drafts_created{model_name=\"m\"} 1.781341706e+09\n"
    )
    m = parse_spec_metrics(txt)
    assert m["num_accepted_tokens"] == 189.0  # not 0.0 from _per_pos, not the _created timestamp
    assert m["num_drafts"] == 161.0
    assert acceptance_length(m["num_accepted_tokens"], m["num_drafts"]) == 2.174


def test_acceptance_length_is_accepted_over_drafts_plus_one():
    # 8000 accepted across 1000 draft steps -> 8 accepted/step, +1 bonus = 9.0 tokens/forward
    assert acceptance_length(8000.0, 1000.0) == 9.0


def test_acceptance_length_baseline_comparable_to_one():
    # zero drafts (no spec) -> undefined, not a crash; a no-spec run advances 1 tok/step
    assert acceptance_length(0.0, 0.0) is None


def test_acceptance_rate_is_accepted_over_proposed():
    assert acceptance_rate(8000.0, 15000.0) == 0.5333
    assert acceptance_rate(5.0, 0.0) is None


def test_speedup_ratio_and_guard():
    assert speedup(484.0, 207.0) == 2.338
    assert speedup(100.0, 0.0) is None


def test_delta_clamps_counter_reset_to_zero():
    before = {"num_accepted_tokens": 8000.0, "num_drafts": 1000.0}
    after = {"num_accepted_tokens": 8400.0, "num_drafts": 1050.0}
    d = delta(before, after)
    assert d["num_accepted_tokens"] == 400.0 and d["num_drafts"] == 50.0
    # server restarted between snapshots -> counters went backwards -> clamp, don't go negative
    assert delta({"num_drafts": 9.0}, {"num_drafts": 2.0})["num_drafts"] == 0.0


def test_summarize_acceptance_window():
    d = delta(parse_spec_metrics(""), parse_spec_metrics(SAMPLE))
    s = summarize_acceptance(d)
    assert s["acceptance_length"] == 9.0
    assert s["acceptance_rate"] == 0.5333
    assert s["accepted_tokens"] == 8000.0


def test_summarize_acceptance_empty_is_all_none_not_crash():
    s = summarize_acceptance({})
    assert s["acceptance_length"] is None and s["acceptance_rate"] is None
