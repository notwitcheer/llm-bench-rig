---
id: ADR-0002
title: Every benchmark result records think mode
status: accepted
date: 2026-06-04
enforced_by: tools/adr/check_think_recorded.py
---
## Context
A thinking-on (reasoning) model's quality is not comparable to a thinking-off
model's — comparing them on one axis is meaningless. The quality leaderboard is
split by think mode, which only works if every result is labeled with the mode it
was run under.

## Decision
`bench.run_benchmark` must record `meta["think"]` (from `quality.think`) before
calling `save_metadata`, so no result is ever saved think-mode-unlabeled.

## Consequences
`build_quality_board` can group results by `meta["think"]`; the split leaderboard
and the dataset card stay correct without manual backfill.

## Enforcement
`tools/adr/check_think_recorded.py` AST-walks `run_benchmark` and fails if the
`meta["think"]` assignment is missing, if `save_metadata` is never called, or if
the assignment is ordered after `save_metadata`. Run: `python -m tools.adr.run_all`.
