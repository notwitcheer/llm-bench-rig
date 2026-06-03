from cadence.threshold import parse_judge_output, decide

GOOD = {"criteria": {
    "ev_plus": {"score": 0.9, "reason": "adds the mechanism"},
    "sourced": {"score": 0.85, "reason": "numbers + repro link"},
    "finding": {"score": 0.8, "reason": "genuine per-model insight"},
    "voice":   {"score": 0.9, "reason": "thesis-first, in register"},
    "meta":    {"score": 0.85, "reason": "a builder would bookmark"}}}

def test_decide_all_high_passes():
    d = decide(GOOD)
    assert d["pass"] is True and d["valid"] is True
    assert d["overall"] == 0.86 and d["fail_reasons"] == []

def test_decide_meta_below_threshold_fails():
    j = {"criteria": {**GOOD["criteria"], "meta": {"score": 0.6, "reason": "forgettable"}}}
    d = decide(j)
    assert d["pass"] is False
    assert any("meta" in r for r in d["fail_reasons"])

def test_decide_one_criterion_below_fails():
    j = {"criteria": {**GOOD["criteria"], "ev_plus": {"score": 0.65, "reason": "restates headline"}}}
    assert decide(j)["pass"] is False

def test_decide_missing_criterion_is_invalid():
    j = {"criteria": {"ev_plus": {"score": 0.9, "reason": "x"}}}
    d = decide(j)
    assert d["pass"] is False and d["valid"] is False

def test_parse_fenced_json():
    txt = 'here is my eval:\n```json\n{"criteria": {"meta": {"score": 0.8, "reason": "ok"}}}\n```\ndone'
    assert parse_judge_output(txt)["criteria"]["meta"]["score"] == 0.8

def test_parse_garbage_returns_empty():
    assert parse_judge_output("the model rambled with no json") == {}
