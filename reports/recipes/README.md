# Recipes: which llama-server line for this model on a 5090

One page per model on the board, one recommended line at the top of each, and the six-set table it came from. Every set is served and measured the same way (`scripts/recipe_bench.py`: four short workloads plus a 16k-token system prompt cold and warm, p50 over 8 requests, VRAM peak, a 40-item GPQA-diamond spot check on the KV-quantised sets). The sets are always: base, `--flash-attn on`, + KV `q8_0`, + KV `q4_0`, + `--parallel 1`, + the model's MTP draft head when one exists. Method in full at the bottom of every page.

| model | the line, in one phrase | short tok/s | on a 16k prefix | VRAM (32k ctx) | page |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B Q6_K | `-fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2` (embedded head) | 144 code / 111 prose | 132 | 25.1 GiB | [recipe](qwen3-8-27b-q6-k.md) |
| Gemma 4 31B Q4_0 (QAT) | `-fa on -np 1`, add the MTP head for code | 76 (130 code with head) | 69 (95 with head) | 20.9 GiB (23.8) | [recipe](gemma-4-31b-q4-0.md) |

![pair 1 chart](../recipes-pair1.png)

## What held across both models (2 of 6 pages in)

- **`--flash-attn on` changes nothing measurable** on this card and build (auto already picks it). Keep it explicit for hygiene.
- **`--parallel 1` is free and sometimes a gain.** Nothing on Qwen; +10% at 16k depth and 2.3 GiB saved on Gemma, whose sliding-window layers care about the cache layout. Always set it on a single-user box.
- **q8_0 KV cache is the memory lever, q4_0 KV is a speed tax at depth.** q8 costs 1 to 5 tok/s and saves 0.9 to 2.6 GiB at 32k ctx. q4 saves 1.4 to 4.1 GiB and costs 17 to 23% of decode against a 16k prefix on both models. The 40-item GPQA spot checks did not move by more than 2 items on either set, so at this sample the cost is speed, not accuracy.
- **MTP is model-dependent.** Qwen's embedded head accepts 0.61 to 0.99 of its drafts and pays 1.8 to 2.3x; Gemma's community head accepts 0.36 to 0.51 and pays 1.5 to 1.7x, code first.
- **Cold TTFT on 16k tokens is 5.0 to 5.8 s on a dense 27B/31B; warm is 0.12 to 0.37 s.** Prompt caching is the flag that matters most and it is on by default (`cache_prompt`).

Queued: Qwen3.6-35B-A3B UD-Q5_K_M, Nemotron 3.5 Lightning Q4_K_M (+ MTP head), Ornith 1.5 35B Q4_K_M, Qwen3.8-Flash-Next UD-Q2_K_XL (offload line).
