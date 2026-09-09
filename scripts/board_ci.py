#!/usr/bin/env python3
"""Walk a results root and write dataset/board_ci.csv with Wilson 95% intervals.

Columns: slug,task,score,correct,total,parse_failures,ci_low,ci_high,
         tokens_per_correct,completion_tokens_median,capped_rate

One row per task that has a detail json (the five board tasks plus gpqa), and one
`q_avg` row per slug whose five board tasks are all present; its ci_low/ci_high
come from the propagated half-width (see lib/ci.py). Only slugs that have a
quality.json are included by default, since those are the rows the board shows;
pass --all to include partial or private dirs as well. Think-on gpqa dirs
(`<slug>-thinkon/`, or gpqa.json with `regime: think-on`) are the exception: their
gpqa.json alone yields a `gpqa_thinkon` row, and the last three columns carry the
tokens-per-correct standing metric; every other row leaves them empty.

Usage:
    python3 scripts/board_ci.py results/ [-o dataset/board_ci.csv] [--all]

Prints the median and maximum q_avg half-width and the task that dominates the
variance, so the number quoted in dataset/README.md is measured, not guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.ci import BOARD_TASKS, dominant_variance_task, slug_intervals  # noqa: E402

COLUMNS = ["slug", "task", "score", "correct", "total", "parse_failures", "ci_low", "ci_high",
           "tokens_per_correct", "completion_tokens_median", "capped_rate"]
# the three token columns are only ever filled on gpqa_thinkon rows
TOKEN_COLUMNS = ("tokens_per_correct", "completion_tokens_median", "capped_rate")
THINKON_SUFFIX = "-thinkon"


def _fmt(v):
    return "" if v is None else v


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text() or "null")
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def is_thinkon_dir(d: Path, gpqa: dict | None = None) -> bool:
    """A think-on gpqa result: `<slug>-thinkon/` or gpqa.json saying so itself."""
    if d.name.endswith(THINKON_SUFFIX):
        return True
    if gpqa is None:
        gpqa = _load_json(d / "gpqa.json")
    if not gpqa:
        return False
    return gpqa.get("regime") == "think-on" or gpqa.get("think") is True


def token_extras(d: Path, gpqa: dict | None) -> dict:
    """The three token columns for a gpqa_thinkon row.

    Prefers the keys GPQAEval now writes into gpqa.json. Older think-on dirs only
    carry the hand-rolled gpqa_tokens.jsonl (one {completion_tokens, ...} per item,
    no correctness), so median and tokens_per_correct (sidecar total / gpqa.json
    correct) are derived from it; capped_rate needs a per-row `capped` flag or a
    `max_tokens` budget and stays empty otherwise.
    """
    gpqa = gpqa or {}
    out = {c: gpqa.get(c) for c in TOKEN_COLUMNS}
    if all(out[c] is not None for c in TOKEN_COLUMNS):
        return out
    side = d / "gpqa_tokens.jsonl"
    if not side.exists():
        return out
    rows = []
    for line in side.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and isinstance(r.get("completion_tokens"), int):
            rows.append(r)
    if not rows:
        return out
    tokens = [r["completion_tokens"] for r in rows]
    if out["completion_tokens_median"] is None:
        out["completion_tokens_median"] = statistics.median(tokens)
    correct = gpqa.get("correct")
    if out["tokens_per_correct"] is None and isinstance(correct, int) and correct > 0:
        out["tokens_per_correct"] = round(sum(tokens) / correct, 1)
    if out["capped_rate"] is None:
        budget = gpqa.get("max_tokens")
        if all("capped" in r for r in rows):
            capped = sum(1 for r in rows if r["capped"])
        elif isinstance(budget, int):
            capped = sum(1 for t in tokens if t >= budget - 8)
        else:
            return out
        out["capped_rate"] = round(capped / len(rows), 4)
    return out


def collect_rows(results_root: Path, include_all: bool = False) -> tuple[list[dict], list[dict]]:
    """Return (csv_rows, per_slug_summaries) for every eligible slug directory.

    Think-on gpqa dirs (`<slug>-thinkon/`, or a gpqa.json with regime think-on)
    never carry a quality.json, so they are eligible on their gpqa.json alone; their
    gpqa row is emitted as task `gpqa_thinkon` (slug stays the dir name) with the
    three token columns filled. Other rows leave those columns empty.
    """
    rows, summaries = [], []
    for d in sorted(p for p in Path(results_root).iterdir() if p.is_dir()):
        gpqa = _load_json(d / "gpqa.json")
        thinkon = is_thinkon_dir(d, gpqa)
        if not include_all and not (d / "quality.json").exists() and not (thinkon and gpqa):
            continue
        res = slug_intervals(d)
        if not res["tasks"]:
            continue
        slug = res["slug"]
        for t in list(BOARD_TASKS) + ["gpqa"]:
            r = res["tasks"].get(t)
            if r is None:
                continue
            row = {"slug": slug, "task": t, "score": r["score"],
                   "correct": r["correct"], "total": r["total"],
                   "parse_failures": _fmt(r["parse_failures"]),
                   "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
            row.update({c: "" for c in TOKEN_COLUMNS})
            if t == "gpqa" and thinkon:
                row["task"] = "gpqa_thinkon"
                row.update({c: _fmt(v) for c, v in token_extras(d, gpqa).items()})
            rows.append(row)
        q = res["q_avg"]
        if q is not None:
            rows.append({"slug": slug, "task": "q_avg", "score": q["score"],
                         "correct": q["correct"], "total": q["total"],
                         "parse_failures": _fmt(q["parse_failures"]),
                         "ci_low": q["ci_low"], "ci_high": q["ci_high"],
                         **{c: "" for c in TOKEN_COLUMNS}})
            dom = dominant_variance_task(res["tasks"])
            summaries.append({"slug": res["slug"], "halfwidth": q["halfwidth"],
                              "dominant": dom[0] if dom else None,
                              "dominant_share": dom[1] if dom else None})
    return rows, summaries


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(summaries: list[dict]) -> str:
    if not summaries:
        return "no complete five-task slugs found"
    hws = [s["halfwidth"] for s in summaries]
    doms = Counter(s["dominant"] for s in summaries)
    shares = [s["dominant_share"] for s in summaries if s["dominant_share"] is not None]
    top, n = doms.most_common(1)[0]
    lines = [
        f"q_avg rows: {len(summaries)}",
        f"q_avg 95% half-width: median {statistics.median(hws):.2f}, "
        f"min {min(hws):.2f}, max {max(hws):.2f} pts",
        f"dominant variance task: {top} in {n}/{len(summaries)} rows, "
        f"median share {statistics.median(shares):.0%}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results_root")
    ap.add_argument("-o", "--out", default="dataset/board_ci.csv")
    ap.add_argument("--all", action="store_true", help="include dirs without quality.json")
    args = ap.parse_args(argv)
    rows, summaries = collect_rows(Path(args.results_root), include_all=args.all)
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")
    print(summarise(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
