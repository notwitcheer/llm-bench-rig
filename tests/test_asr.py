"""ASR head-to-head core: normalizer + WER + split scoring. Stdlib only, Mac, no GPU."""
from lib.asr.normalize import normalize


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_normalize_drops_apostrophes_into_one_token():
    assert normalize("don't") == "dont"
    assert normalize("It's 5 o'clock.") == "its 5 oclock"


def test_normalize_collapses_whitespace_and_handles_empty():
    assert normalize("  a   b\tc \n") == "a b c"
    assert normalize("") == ""
    assert normalize(None) == ""


def test_normalize_unicode_apostrophe():
    assert normalize("don’t") == "dont"


from lib.asr.wer import wer, _edit_distance


def test_edit_distance_basic():
    assert _edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert _edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1   # substitution
    assert _edit_distance(["a", "b"], ["a", "b", "c"]) == 1        # insertion
    assert _edit_distance(["a", "b", "c"], ["a", "c"]) == 1        # deletion


def test_wer_identical_is_zero():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_wer_one_sub_in_four_words():
    assert wer("the quick brown fox", "the quick red fox") == 0.25


def test_wer_normalizes_before_scoring():
    assert wer("Hello, World!", "hello world") == 0.0


def test_wer_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "spurious words") == 1.0


from lib.asr.score import score_split, rtfx


def test_score_split_micro_averages_wer():
    pairs = [("the quick brown fox", "the quick red fox"), ("good day", "good day")]
    out = score_split(pairs)
    assert out["n"] == 2
    assert out["words"] == 6
    assert out["wer"] == round(1 / 6, 4)


def test_score_split_empty():
    assert score_split([]) == {"wer": 0.0, "n": 0, "words": 0}


def test_rtfx():
    assert rtfx(600.0, 30.0) == 20.0
    assert rtfx(100.0, 0.0) == 0.0


import json
from scripts.asr_report import summarize_model


def test_summarize_model_computes_per_split_wer_and_rtfx(tmp_path):
    d = tmp_path / "parakeet"; d.mkdir()
    rows = [
        {"id": "1", "split": "test-clean", "reference": "the quick brown fox", "hypothesis": "the quick red fox"},
        {"id": "2", "split": "test-clean", "reference": "good day", "hypothesis": "good day"},
        {"id": "3", "split": "test-other", "reference": "hello there world", "hypothesis": "hello there world"},
    ]
    (d / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "timing.json").write_text(json.dumps({"audio_seconds": 600.0, "proc_seconds": 30.0, "peak_vram_mib": 4200}))

    s = summarize_model(str(d))
    assert s["wer_clean"] == round(1 / 6, 4)
    assert s["wer_other"] == 0.0
    assert s["rtfx"] == 20.0
    assert s["peak_vram_mib"] == 4200


from scripts.chart_asr import label_for


def test_label_for_formats_model_line():
    s = {"model": "parakeet-tdt-0.6b-v2", "wer_clean": 0.062, "wer_other": 0.094, "rtfx": 350.0}
    assert label_for(s) == "parakeet-tdt-0.6b-v2\nWER 6.2/9.4%  ·  350x"
