#!/usr/bin/env python3
"""Walk a results root and write dataset/board_ci.csv with Wilson 95% intervals.

Columns: slug,task,score,correct,total,parse_failures,ci_low,ci_high

One row per task that has a detail json (the five board tasks plus gpqa), and one
`q_avg` row per slug whose five board tasks are all present; its ci_low/ci_high
come from the propagated half-width (see lib/ci.py). Only slugs that have a
quality.json are included by default, since those are the rows the board shows;
pass --all to include partial or private dirs as well.

Usage:
    python3 scripts/board_ci.py results/ [-o dataset/board_ci.csv] [--all]

Prints the median and maximum q_avg half-width and the task that dominates the
variance, so the number quoted in dataset/README.md is measured, not guessed.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.ci import BOARD_TASKS, dominant_variance_task, slug_intervals  # noqa: E402

COLUMNS = ["slug", "task", "score", "correct", "total", "parse_failures", "ci_low", "ci_high"]


def _fmt(v):
    return "" if v is None else v


def collect_rows(results_root: Path, include_all: bool = False) -> tuple[list[dict], list[dict]]:
    """Return (csv_rows, per_slug_summaries) for every eligible slug directory."""
    rows, summaries = [], []
    for d in sorted(p for p in Path(results_root).iterdir() if p.is_dir()):
        if not include_all and not (d / "quality.json").exists():
            continue
        res = slug_intervals(d)
        if not res["tasks"]:
            continue
        for t in list(BOARD_TASKS) + ["gpqa"]:
            r = res["tasks"].get(t)
            if r is None:
                continue
            rows.append({"slug": res["slug"], "task": t, "score": r["score"],
                         "correct": r["correct"], "total": r["total"],
                         "parse_failures": _fmt(r["parse_failures"]),
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        q = res["q_avg"]
        if q is not None:
            rows.append({"slug": res["slug"], "task": "q_avg", "score": q["score"],
                         "correct": q["correct"], "total": q["total"],
                         "parse_failures": _fmt(q["parse_failures"]),
                         "ci_low": q["ci_low"], "ci_high": q["ci_high"]})
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
