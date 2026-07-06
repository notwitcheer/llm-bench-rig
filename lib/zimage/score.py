"""Image-gen speed metrics.

Throughput for few-step text-to-image: seconds per image (compute) and images
per minute. ``gen_seconds`` = the cuda-synchronized pipeline call (pure
diffusion + VAE compute); ``wall_seconds`` = incl. PNG save (the honest
end-user number). Kept modality-agnostic in spirit with ``lib.ace_step.score``
(music) and ``lib.asr.score`` (speech). Invariants live here, tested — ADR-0003.
"""
from statistics import mean, pstdev


def images_per_min(gen_seconds: float) -> float:
    """Images generated per minute of compute. 0.0 if gen_seconds<=0 (safe)."""
    if gen_seconds <= 0:
        return 0.0
    return round(60.0 / gen_seconds, 2)


def megapixels(width: int, height: int) -> float:
    """Image size in megapixels (width*height / 1e6), rounded to 3dp."""
    return round(width * height / 1_000_000, 3)


def aggregate(records: list) -> dict:
    """Group per-image records into ``(resolution, steps)`` cells with stats.

    Each record must carry ``resolution`` (int square side), ``steps`` and
    ``gen_seconds``; ``wall_seconds`` and ``peak_vram_mib`` are optional
    (default to ``gen_seconds`` and 0). Returns a dict keyed by
    ``(resolution, steps)`` -> per-cell means/stds + max VRAM + images/min.
    Single-record cells report std 0.0.
    """
    cells: dict = {}
    for r in records:
        cells.setdefault((r["resolution"], r["steps"]), []).append(r)
    out: dict = {}
    for key, recs in cells.items():
        gens = [r["gen_seconds"] for r in recs]
        walls = [r.get("wall_seconds", r["gen_seconds"]) for r in recs]
        vrams = [r.get("peak_vram_mib", 0) for r in recs]
        out[key] = {
            "n": len(recs),
            "gen_seconds_mean": round(mean(gens), 3),
            "gen_seconds_std": round(pstdev(gens), 3) if len(gens) > 1 else 0.0,
            "wall_seconds_mean": round(mean(walls), 3),
            "images_per_min": images_per_min(mean(gens)),
            "peak_vram_mib_max": max(vrams),
        }
    return out
