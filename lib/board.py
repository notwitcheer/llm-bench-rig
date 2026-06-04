"""Quality leaderboard helpers — think-mode grouping + q_avg, split by reasoning regime.

Comparing a think-ON (reasoning) model's quality against a think-OFF model is
apples-to-oranges, so the quality board is split into two groups. Think mode is
recorded per result in meta.json ("think": true/false); these sets are the
verified ground truth (from each model's run report **Thinking:** line) used to
backfill existing results and to classify anything meta.json doesn't yet carry.
"""
from pathlib import Path
import json

# Verified against each model's reports/<slug>.md "**Thinking:**" line (2026-06-04).
# gpt-oss-20b-q4-k-m had no explicit line but ran think-OFF: it was part of the
# think-off field and is the textbook inline-reasoning-with-/nothink case that the
# HumanEval stop-sequence bug truncated.
THINK_OFF = frozenset({
    "gemma-4-12b-it-q6-k",
    "gemma-4-31b-it-q6-k",
    "gpt-oss-20b-q4-k-m",
    "nvidia-nemotron-cascade-2-30b-a3b-q4-k-m",
    "qwen3-6-27b-q6-k",
    "qwen3-6-27b-q4-k-m",
    "qwen3-6-27b-mtp-q4-k-m",
    "qwen3-6-35b-a3b-ud-q4-k-m",
    "qwen3-coder-next-ud-q2-k-xl",
})
THINK_ON = frozenset({
    "gpt-oss-120b-mxfp4",
    "qwen3-6-28b-reap20-a3b-q6-k",
    "qwen3-6-35b-a3b-ud-q6-k",
})

QUALITY_TASKS = ("mmlu", "arc_challenge", "hellaswag", "humaneval", "gsm8k")


def _score(val):
    """A quality entry is either {"score": x, ...} or a bare number."""
    if isinstance(val, dict):
        return val.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def quality_average(quality: dict) -> float | None:
    """Mean of the present numeric task scores. None if nothing scorable."""
    vals = [s for s in (_score(quality.get(t)) for t in QUALITY_TASKS) if isinstance(s, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def think_mode(meta: dict, slug: str | None = None) -> bool | None:
    """True=ON, False=OFF, None=unknown. meta['think'] wins; else fall back to the
    verified sets so an un-backfilled result still classifies."""
    if isinstance(meta, dict) and isinstance(meta.get("think"), bool):
        return meta["think"]
    slug = slug or (meta.get("slug") if isinstance(meta, dict) else None)
    if slug in THINK_ON:
        return True
    if slug in THINK_OFF:
        return False
    return None


def build_quality_board(results_dir) -> dict:
    """Scan results/, return {'on': [...], 'off': [...], 'unknown': [...]} where each
    entry is {slug, name, think, q_avg, scores{task->score}}. Each group sorted by
    q_avg desc. 'unknown' = a result whose think mode can't be determined (surfaced,
    never silently dropped)."""
    results_dir = Path(results_dir)
    on, off, unknown = [], [], []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        mf, qf = d / "meta.json", d / "quality.json"
        if not (mf.exists() and qf.exists()):
            continue
        meta = json.loads(mf.read_text())
        quality = json.loads(qf.read_text())
        qavg = quality_average(quality)
        if qavg is None:
            continue
        tm = think_mode(meta, d.name)
        entry = {
            "slug": d.name,
            "name": meta.get("name", d.name),
            "quant": meta.get("quant", ""),
            "think": tm,
            "q_avg": qavg,
            "scores": {t: _score(quality.get(t)) for t in QUALITY_TASKS},
        }
        (on if tm is True else off if tm is False else unknown).append(entry)
    for group in (on, off, unknown):
        group.sort(key=lambda e: e["q_avg"], reverse=True)
    return {"on": on, "off": off, "unknown": unknown}
