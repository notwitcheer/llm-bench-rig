"""Corpus-level scoring. WER is MICRO-averaged (total errors / total reference words),
the standard for LibriSpeech — not the mean of per-utterance WERs. RTFx = audio seconds
processed per wall-clock second (higher = faster)."""
from lib.asr.normalize import normalize
from lib.asr.wer import _edit_distance


def score_split(pairs: list) -> dict:
    """pairs: list of (reference, hypothesis). Returns micro-averaged WER over all pairs."""
    total_words = 0
    total_errs = 0
    for ref, hyp in pairs:
        r = normalize(ref).split()
        total_words += len(r)
        total_errs += _edit_distance(r, normalize(hyp).split())
    return {"wer": round(total_errs / total_words, 4) if total_words else 0.0,
            "n": len(pairs), "words": total_words}


def rtfx(audio_seconds: float, proc_seconds: float) -> float:
    return round(audio_seconds / proc_seconds, 1) if proc_seconds > 0 else 0.0
