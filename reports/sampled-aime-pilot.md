# Greedy against sampled on AIME 2026: eight samples per problem move a 30-problem score by 2.5 and 3.75 points

**TL;DR.** The boards here score one greedy pass per item. DeepSeek's DeepInfer eval reports AIME 2026 as a mean over many samples per problem (32 on AIME, 8 on GPQA Diamond; `eval/README.md`, read 2026-10-20). This pilot reruns AIME 2026 (30 problems) on the two rows of the 2026-10-20 AIME tie, thinking on at the 32k budget, with **8 sampled completions per problem** (temperature 0.6, top-p 0.95, top-k 20, the Qwen3.8 card's thinking-mode settings), same prompt, grader and exact-number veto as the greedy rows.

| row | greedy (1 pass) | sampled mean@8 | the 8 runs, min to max | pass@8 | majority of 8 | capped, greedy / sampled |
|---|---:|---:|---|---:|---:|---|
| Qwen3.8-27B Q6_K | 83.33 (25/30) | **85.83** (206/240) | 76.67 to 93.33, sd 5.71 | 28/30 | 27/30 | 7/30 (23.3%) / 42/240 (17.5%) |
| Ternary-Bonsai-2-27B PQ2_0 | 80.00 (24/30) | **83.75** (201/240) | 80.00 to 86.67, sd 2.60 | 28/30 | 26/30 | 7/30 (23.3%) / 46/240 (19.2%) |

1. **Greedy sits inside the spread of the sampled runs.** Q6_K's eight runs span 76.67 to 93.33 (a 16.7-point range from one model, one prompt set); its greedy 83.33 is above three runs and below five. Mean@8 is 2.5 points above greedy for Q6_K and 3.75 for PQ2_0. Resampling problems (30 of them, the real unit of independence), the paired greedy-minus-sampled difference is −2.5 ± 8.0 points (95%) for Q6_K and −3.75 ± 11.4 for PQ2_0. Neither is a gap this set can see.
2. **The tie stays a tie.** Q6_K minus PQ2_0 is +2.08 points on mean@8 (paired over the 30 problems, ± 4.08 at 95%), against +3.33 greedy (25 vs 24 problems is one problem). Sampling does not break the 2026-10-20 tie in either direction.
3. **Where greedy loses, it hits the budget.** Every problem greedy misses is also a cap hit at 32,768 tokens (Q6_K: 5 misses, all capped; PQ2_0: 6 misses, all capped). PQ2_0 problems 10 and 17 are solved 8/8 when sampled and capped under greedy, so a greedy non-termination, not a wrong answer, costs it two problems. Sampling trims the capped share (23.3% greedy; 17.5% and 19.2% sampled) but does not remove it: problems 11 and 15 are 0/8 on both models, all 16 samples capped, so they are budget floors, not hard problems.
4. **Mean@8 does not buy a narrow interval on 30 problems.** The problem-level bootstrap 95% interval on mean@8 is 75.8 to 94.2 (Q6_K) and 72.1 to 93.3 (PQ2_0): about ±9 points, like the single-pass Wilson interval it replaces. What 8 samples add is the run-to-run spread (sd 5.7 and 2.6 points of one pass) and a pass@8 view: 28 of 30 on both models.

## Setup

- **Task:** `MathArena/aime_2026`, 30 problems, published 2026-02-13. Thinking on, `max_tokens 32768`, `ctx 40960` per slot (the greedy rows' regime), zero-shot, grader: vendored Qwen2.5-Math grade plus the exact-number veto of `lib/evals/matharena.py` (0 vetoes, 0 grade timeouts).
- **Sampling:** temperature 0.6, top-p 0.95, top-k 20, K = 8 completions per problem (240 samples per row). A per-(problem, sample) seed is recorded but llama.cpp parallel slots are not batch-invariant, so a rerun need not reproduce the same tokens.
- **Driver:** `scripts/sampled_contest.py` (resumable per sample; one llama-server with `-np` slots, one request per slot in flight, slots are independent sequences so each sample sees the full context). Queue item `scripts/capsule-queue/67c-sampled-aime-pilot.sh`, concurrency probed first: Q6_K `-np 3` 162.3 tok/s aggregate (one stream 62.4, 29.4 GB; `-np 4` OOMs on the first concurrent batch), PQ2_0 `-np 8` 457.4 tok/s (one stream 141.7, 29.1 GB).
- **Servers:** Q6_K on llama.cpp `b10371` (the rig's mainline build); PQ2_0 on the PrismML fork `4fbda12` (the packing is not in mainline), `-np 8`.
- **Run window:** PQ2_0 finished Friday 2026-10-23 at 06:05 CEST and Q6_K Saturday 2026-10-24 at 07:05 CEST, both resumed across heures-creuses blocks (Q6_K's last block ran 273 minutes). The driver checkpoints after every finished sample, so a hard stop lost only samples in flight.
- **Greedy comparators:** `results/qwen3-8-27b-q6-k-matharena-32k` (25/30) and `results/ternary-bonsai-2-27b-pq2-0-matharena-32k` (24/30), published in [`matharena-contests-32k.md`](matharena-contests-32k.md); same prompt, grader and budget.

## Per-problem picture

Out of 30 problems: Q6_K solves 17 of them 8/8, 11 on some samples, 2 never; PQ2_0 solves 19 8/8, 9 on some, 2 never. The two never-solved problems (11 and 15) are the same for both models and capped on all 16 samples. The unstable ones are the late, long problems: Q6_K problem 28 (7/8 solved, 6 of 8 samples capped), 29 (5/8, 4 capped) and 30 (3/8, 7 capped); PQ2_0 problem 28 (5/8, 6 capped), 29 (3/8, 6 capped), 30 (1/8, 8 capped). Greedy misses on problems the sampler solves most of the time: Q6_K 7 (7/8) and 29 (5/8); PQ2_0 10 (8/8), 17 (8/8), 28 (5/8).

## Worth it?

**Worth doing if** you report one number for a 30-problem contest set: the single greedy pass is a draw from a range that is 10 to 17 points wide, and the honest statement is a mean of several samples plus the range. **Not worth it** as a way to separate two close models on AIME alone: 8 samples reduce the variance of the draw, not the 30-problem noise between problems, so the intervals on mean@8 stay about ±9 points.

## Honest limits

- A pilot: two rows, one contest set (AIME 2026, 30 problems), K = 8 rather than DeepInfer's 32. HMMT and the other seven second-tier rows were not run sampled.
- Sampling settings are the Qwen3.8 card's; PQ2_0 is a derivative checkpoint and inherits them, which is an assumption.
- llama.cpp parallel slots are not batch-invariant: a rerun of the same seeds will not reproduce these tokens exactly.
- The two rows ran on different servers (mainline and the Prism fork) at different slot counts, as the greedy rows did.
- The budget floor (problems 11 and 15 capped on every sample, 16 of 16) means mean@8 is a floor on a model that would solve them with more tokens; a 64k pass was not run for this study.
- Thinking-on only; greedy rows were run with one pass, so their own variance is unmeasured by design.

## Files

- Raw per-sample records (`<task>_samples.json`) and the summary (`<task>_sampled_detail.json`) are in `results/<slug>-matharena-32k-s8/` on the capsule.
- Driver: `scripts/sampled_contest.py`. Queue item: `scripts/capsule-queue/67c-sampled-aime-pilot.sh`.
