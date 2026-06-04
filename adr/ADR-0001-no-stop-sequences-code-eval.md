---
id: ADR-0001
title: No stop sequences in the code-eval harness
status: accepted
date: 2026-06-04
enforced_by: tools/adr/check_no_stop_sequences.py
---
## Context
The HumanEval harness passed API stop sequences (`\ndef `, `\nclass `) to the
model. Reasoning models reason *inline* and write "def"/"class" mid-reasoning, so
the stop fired before any code was emitted — producing empty/truncated responses
and false-low scores (Qwen3-Coder-Next read 10%, not 93%). Discovered and
corrected in the cp8 field correction, 2026-06-04.

## Decision
The code-evaluation harness (`lib/evals/humaneval.py`) must never pass a `stop=`
argument to a model call. Truncation of trailing extra functions happens in
post-processing on the captured response, not via generation-time stops.

## Consequences
Reasoning models get the full token budget to reason then emit code. HumanEval is
trustworthy across reasoning and non-reasoning models.

## Enforcement
`tools/adr/check_no_stop_sequences.py` AST-walks the harness and fails if any call
carries a `stop=` keyword. Run: `python -m tools.adr.run_all`. A violation prints
`ADR-0001 VIOLATION: stop= passed to a call at <file>:<line> …` and blocks the commit.
