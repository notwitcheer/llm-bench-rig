"""Aggregate per-utterance TTS metrics into per-model means + spread. Pure stdlib. None values
(e.g. a missing SIM-o) are dropped from that metric's mean."""
import statistics as st


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(per_utt):
    out = {"n": len(per_utt)}
    for k in ("wer", "rtfx", "first_audio_s", "sim"):
        vals = [u.get(k) for u in per_utt]
        out[f"{k}_mean"] = _mean(vals)
        out[f"{k}_std"] = _std(vals)
    return out
