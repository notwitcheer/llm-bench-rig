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
