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
