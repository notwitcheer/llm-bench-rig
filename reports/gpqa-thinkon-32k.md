# GPQA Diamond at a 32k thinking budget: the three rows that capped at 16k

**Rig:** RTX 5090 32GB (capsule), llama.cpp build 10371 (`5d16e81dd`), llama-server, batch 1 · **Window:** 2026-10-03 to 2026-10-04, inside the box's cheap-electricity hours · **Harness:** [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig), `lib/evals/gpqa.py`, idle-queue item `60-gpqa-thinkon-32k`, artefacts in `results/<slug>-thinkon-32k/` · **Data:** [`dataset/gpqa_thinkon_32k.csv`](../dataset/gpqa_thinkon_32k.csv) · **Sibling:** [MATH-500 at a 32k budget](second-tier-thinkon-32k.md)

The same check as the MATH-500 32k pass, on a second test. GPQA Diamond is 198 graduate-level multiple-choice questions in biology, physics and chemistry. At the think-on regime of `max_tokens 16384`, three rows hit the completion cap on more than 10% of the questions. This pass gives exactly those rows 32,768 tokens and changes nothing else that affects the answers.

## TL;DR

1. **Doubling the budget added 3.0 to 7.1 points.** Qwable-27B 78.28 to 85.35, Nemotron 3.5 Lightning 30B-A3B 60.61 to 67.17, Qwen3.6-27B 76.77 to 79.80.
2. **Every answer that finished inside 16k got the same verdict and the same completion length at 32k** (124/124, 153/153, 167/167). Every added point is a question that had hit the 16k cap.
3. **Nemotron 3.5 Lightning still caps on 19.7% at 32k**, so its 67.17 is a floor, not a ceiling. The other two cap at 6.6% and 5.1%.

## Setup

- **Regime:** thinking on, `max_tokens 32768`, `ctx 40960`, greedy, zero-shot, one pass per question. The 16k cells used `max_tokens 16384`, `ctx 24576`. Each 32k gpqa.json carries `bench` `gpqa-thinkon-32k (separate higher-budget pass over rows capped >10% at 16k)`.
- **Task:** GPQA Diamond, 198 questions (`Idavidrein/gpqa`, diamond subset), letter answer.
- **Rows:** the same three GGUF files as the 16k think-on cells.

## The table

| Model | Quant | GGUF | 16k | 16k cap | 32k | 32k cap | 32k tokens / correct |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwable-27B | Q4_K_M | 16.5 GB | 78.28 | 22.7% | **85.35** | 6.6% | 11,737 |
| Qwen3.6-27B | Q6_K | 22.9 GB | 76.77 | 15.7% | **79.80** | 5.1% | 11,108 |
| Nemotron-3.5-Lightning 30B-A3B | Q4_K_M | 24.5 GB | 60.61 | 37.4% | **67.17** | 19.7% ▲ | 19,701 |

▲ 10% or more of the questions still hit the 32k cap: a floor, not a ceiling. Per-row `correct`/`total`, `capped_count`, `parse_failures`, `reasoning_fallback_count`, `tokens_per_correct` and the median completion length at both budgets: [`dataset/gpqa_thinkon_32k.csv`](../dataset/gpqa_thinkon_32k.csv).

## Question by question

| Model | finished at 16k | same verdict and length at 32k | hit the 16k cap | of those, correct at 32k | still capped at 32k |
|---|---:|---:|---:|---:|---:|
| Qwable-27B | 153 | 153 | 45 | 30 | 13 |
| Qwen3.6-27B | 167 | 167 | 31 | 15 | 10 |
| Nemotron-3.5-Lightning 30B-A3B | 124 | 124 | 74 | 26 | 39 |

## Reads

1. **The 16k GPQA cells for these rows measured the budget.** Finished answers did not change; what changed was how many answers finished.
2. **The median question is unaffected.** The median completion length is identical at both budgets on all three rows (5,814.5, 5,709 and 7,598 tokens), because the median question finishes well inside 16k.
3. **The same pattern as MATH-500 and this year's contests.** On all three tests, a think-on score below the budget is mostly a count of chains that did not finish ([MATH-500 at 32k](second-tier-thinkon-32k.md), [AIME and HMMT at 32k](matharena-contests-32k.md)).

## Limits

- One greedy pass per question, one card. The Wilson 95% half-width on 198 questions is about 5 to 7 points at these scores, so the Qwen3.6-27B gain (3.0 points, 6 questions) sits inside it; the per-question table above shows where it comes from.
- GPQA Diamond only for these three rows. Rows that capped under 10% at 16k were not rerun.

*Sources: `results/<slug>-thinkon/gpqa.json` (16k) and `results/<slug>-thinkon-32k/gpqa.json` (32k) on the rig; per-question comparison from the matching `gpqa_progress.json` files (`correct`, `capped`, `completion_tokens` per question), read 2026-10-07.*
