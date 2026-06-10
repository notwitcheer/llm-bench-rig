"""The single text normalizer applied to EVERY model's output and the reference before
WER. Identical treatment is the fairness guarantee. Lowercase, drop punctuation, fold
contractions (don't -> dont), collapse whitespace. Number-word vs digit differences are
NOT reconciled (counted as errors, equally for all models) — a stated limitation."""
import re

_KEEP = re.compile(r"[^a-z0-9\s']")  # after lowercasing: keep letters, digits, space, apostrophe


def normalize(text: str) -> str:
    text = (text or "").lower().replace("’", "'")  # unicode apostrophe -> ascii
    text = _KEEP.sub(" ", text)        # punctuation -> space
    text = text.replace("'", "")       # contractions fold to one token: don't -> dont
    return re.sub(r"\s+", " ", text).strip()
