"""Confidence intervals for the quality board.

Every board score is a binomial proportion (items correct out of items scored),
so each carries a Wilson 95% interval computed from the `correct`/`total` pair
that is already in `results/<slug>/<task>_detail.json`. q_avg is the plain mean
of the five task percentages; treating tasks as independent, its variance is the
mean of the task variances divided by the task count, and the half-width reported
here is 1.96 times that standard deviation. Two rows whose q_avg differ by less
than that half-width are a tie, whatever the sort order says.

Nothing here touches the network or a gpu; it reads json files only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from lib.board import QUALITY_TASKS

Z95 = 1.959963984540054
BOARD_TASKS = tuple(QUALITY_TASKS)
# gpqa is the second-tier task: intervals are reported, never folded into q_avg.
CI_TASKS = BOARD_TASKS + ("gpqa",)


def wilson_interval(correct: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1]."""
    if total <= 0:
        raise ValueError("total must be positive")
    if correct < 0 or correct > total:
        raise ValueError("correct must be between 0 and total")
    p = correct / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    lo, hi = centre - half, centre + half
    # centre - half is exactly zero at p=0 (and one at p=1) up to float dust
    if correct == 0:
        lo = 0.0
    if correct == total:
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def wald_variance_pct(correct: int, total: int) -> float:
    """Variance of the score in percentage points squared (p(1-p)/n * 100^2).

    Used only for propagating into q_avg; the per-task interval itself is Wilson.
    """
    if total <= 0:
        return float("nan")
    p = correct / total
    return p * (1 - p) / total * 10000.0


def qavg_halfwidth(task_rows: dict, z: float = Z95) -> float | None:
    """Propagated 95% half-width of the five-task mean, in percentage points.

    `task_rows` maps task -> {"correct", "total"}. Returns None unless all five
    board tasks are present with a positive total.
    """
    variances = []
    for t in BOARD_TASKS:
        r = task_rows.get(t)
        if not r or not r.get("total") or r.get("correct") is None:
            return None
        variances.append(wald_variance_pct(r["correct"], r["total"]))
    var_mean = sum(variances) / (len(variances) ** 2)
    return z * math.sqrt(var_mean)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text() or "null")
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_failures(detail: dict) -> int | None:
    """Top-level parse_failures, or the sum over mmlu's per_subject blocks."""
    if isinstance(detail.get("parse_failures"), int):
        return detail["parse_failures"]
    per_subject = detail.get("per_subject")
    if isinstance(per_subject, dict) and per_subject:
        vals = [s.get("parse_failures") for s in per_subject.values() if isinstance(s, dict)]
        if vals and all(isinstance(v, int) for v in vals):
            return sum(vals)
    return None


def task_row_from_detail(detail: dict) -> dict | None:
    """Extract {score, correct, total, parse_failures, ci_low, ci_high} from one detail json.

    humaneval records `passed` where the others record `correct`. Scores that
    cannot be turned into a count (missing total) give None.
    """
    if not isinstance(detail, dict):
        return None
    total = detail.get("total")
    correct = detail.get("correct")
    if correct is None:
        correct = detail.get("passed")
    if not isinstance(total, int) or total <= 0 or not isinstance(correct, int):
        return None
    lo, hi = wilson_interval(correct, total)
    score = detail.get("score")
    if not isinstance(score, (int, float)):
        score = round(correct / total * 100, 2)
    return {
        "score": float(score),
        "correct": correct,
        "total": total,
        "parse_failures": _parse_failures(detail),
        "ci_low": round(lo * 100, 2),
        "ci_high": round(hi * 100, 2),
    }


def slug_intervals(slug_dir: Path) -> dict:
    """Per-task rows plus the propagated q_avg row for one results/<slug> directory.

    Reads `<task>_detail.json` for the five board tasks, `gpqa_detail.json` or
    `gpqa.json` for the second tier, and `quality.json` for the published scores
    (the detail score wins when both exist and agree; a disagreement is recorded).
    Returns {"slug", "tasks": {task: row}, "q_avg": {...} | None}.
    """
    slug_dir = Path(slug_dir)
    quality = _load_json(slug_dir / "quality.json") or {}
    tasks = {}
    for t in CI_TASKS:
        detail = _load_json(slug_dir / f"{t}_detail.json")
        if detail is None and t == "gpqa":
            detail = _load_json(slug_dir / "gpqa.json")
        row = task_row_from_detail(detail) if detail else None
        if row is None:
            continue
        published = quality.get(t)
        if isinstance(published, dict):
            published = published.get("score")
        if isinstance(published, (int, float)) and abs(published - row["score"]) > 0.05:
            row["published_score"] = float(published)
        tasks[t] = row
    q = None
    hw = qavg_halfwidth(tasks)
    if hw is not None:
        scores = [tasks[t]["score"] for t in BOARD_TASKS]
        mean = sum(scores) / len(scores)
        q = {
            "score": round(mean, 2),
            "halfwidth": round(hw, 2),
            "ci_low": round(mean - hw, 2),
            "ci_high": round(mean + hw, 2),
            "correct": sum(tasks[t]["correct"] for t in BOARD_TASKS),
            "total": sum(tasks[t]["total"] for t in BOARD_TASKS),
            "parse_failures": _sum_or_none(tasks[t]["parse_failures"] for t in BOARD_TASKS),
        }
    return {"slug": slug_dir.name, "tasks": tasks, "q_avg": q}


def _sum_or_none(values) -> int | None:
    vals = list(values)
    if any(v is None for v in vals):
        return None
    return sum(vals)


def dominant_variance_task(task_rows: dict) -> tuple[str, float] | None:
    """Which board task contributes most of the q_avg variance, and its share."""
    parts = {}
    for t in BOARD_TASKS:
        r = task_rows.get(t)
        if not r or not r.get("total"):
            return None
        parts[t] = wald_variance_pct(r["correct"], r["total"])
    total = sum(parts.values())
    if total <= 0:
        return None
    t = max(parts, key=parts.get)
    return t, parts[t] / total
