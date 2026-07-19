---
id: ADR-0004
title: MC evals gate on completion length (pace watchdog)
status: accepted
date: 2026-07-19
enforced_by: tools/adr/check_mc_gate_wired.py
---
## Context
2026-07-16: sglang 0.5.14 GDN-degenerate output fed the MMLU eval. The rambling
contained letters, parse_choice extracted them, and subjects graded at 2-4% — a
catastrophic-quant story that was 100% serving bug; only an unrelated HF Hub 504
stopped the suite from banking it. Parse rate cannot catch this class: degenerate
output still parses. Pace can: think-off MC answers run 1-7 tokens, the poisoned
run averaged ~850.

## Decision
Every multiple-choice eval (parse_choice caller) feeds each answer's completion
token count to a CompletionLengthGate (lib/evals/base.py): rolling window 25,
min 10 samples, threshold `quality.mc_gate_tokens` (default 50, null disables).
The gate raises InstrumentGateError — armed only think-off; think-on runs log the
mean without aborting. run_quality_bench lets it propagate: no results dict, no
banked score. Every run logs `completion_tokens_mean` in the eval's result dict.

## Consequences
- A degenerate server aborts the suite within ~10 answers of onset, even mid-suite.
- Instrument failure is structurally distinct from subject failure (CP43 rule).
- New MC evals cannot ship unwired: check_mc_gate_wired blocks the commit.
