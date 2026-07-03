"""Music-gen speed metrics.

RTF (real-time factor) = seconds of song produced per wall-clock second of
generation. RTF > 1 means the song is written faster than it plays — the
"a home GPU makes a song faster than it plays" number. Reuses
``lib.asr.score.rtfx`` so "real-time factor" means the same thing across
modalities (ASR, TTS, music). Invariants live here, tested — ADR-0003.
"""
from statistics import mean, pstdev

from lib.asr.score import rtfx


def rtf(song_seconds: float, gen_seconds: float) -> float:
    """Real-time factor: song duration / generation wall-clock.

    >1 = faster than playback. Delegates to ``lib.asr.score.rtfx`` (rounds to 1dp).
    """
    return rtfx(song_seconds, gen_seconds)


def aggregate(records: list) -> dict:
    """Group per-song records into (model_tier, duration_s) cells with stats.

    Each record must carry: ``model_tier``, ``duration_s``, ``gen_seconds``,
    ``song_seconds``; ``peak_vram_mib`` optional (defaults 0). Returns a dict
    keyed by ``(model_tier, duration_s)`` → per-cell means/stds + max VRAM.
    Single-record cells report std 0.0.
    """
    cells: dict = {}
    for r in records:
        cells.setdefault((r["model_tier"], r["duration_s"]), []).append(r)
    out: dict = {}
    for key, recs in cells.items():
        gens = [r["gen_seconds"] for r in recs]
        rtfs = [rtf(r["song_seconds"], r["gen_seconds"]) for r in recs]
        vrams = [r.get("peak_vram_mib", 0) for r in recs]
        out[key] = {
            "n": len(recs),
            "gen_seconds_mean": round(mean(gens), 3),
            "gen_seconds_std": round(pstdev(gens), 3) if len(gens) > 1 else 0.0,
            "rtf_mean": round(mean(rtfs), 3),
            "rtf_std": round(pstdev(rtfs), 3) if len(rtfs) > 1 else 0.0,
            "peak_vram_mib_max": max(vrams),
        }
    return out
