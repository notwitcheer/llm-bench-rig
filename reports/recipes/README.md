# Recipes: which llama-server line for this model on a 5090

One page per model on the board, one recommended line at the top of each, and the six-set table it came from. Every set is served and measured the same way (`scripts/recipe_bench.py`: four short workloads plus a 16k-token system prompt cold and warm, p50 over 8 requests, VRAM peak, a 40-item GPQA-diamond spot check on the KV-quantised sets). The sets are always: base, `--flash-attn on`, + KV `q8_0`, + KV `q4_0`, + `--parallel 1`, + the model's MTP draft head when one exists. Method in full at the bottom of every page.

| model | the line, in one phrase | short tok/s | on a 16k prefix | VRAM (32k ctx) | page |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B Q6_K | `-fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2` (embedded head) | 144 code / 111 prose | 132 | 25.1 GiB | [recipe](qwen3-8-27b-q6-k.md) |
| Gemma 4 31B Q4_0 (QAT) | `-fa on -np 1`, add the MTP head for code | 76 (130 code with head) | 69 (95 with head) | 20.9 GiB (23.8) | [recipe](gemma-4-31b-q4-0.md) |
| Qwen3.6-35B-A3B UD-Q5_K_M | `-fa on -np 1`, f16 KV, nothing else moves it | 266 | 248 | 26.1 GiB | [recipe](qwen3-6-35b-a3b-ud-q5-k-m.md) |
| Nemotron 3.5 Lightning 30B-A3B Q4_K_M | `-fa on -np 1`, **no draft head** (MTP is 0.6 to 0.8x here) | 369 | 359 | 23.5 GiB | [recipe](nemotron-3-5-lightning-q4-k-m.md) |
| Ornith 1.5 35B-A3B Q4_K_M | `-fa on -np 1`, f16 KV (q4 KV cost quality here too) | 292 | 270 | 20.8 GiB | [recipe](ornith-1-5-35b-q4-k-m.md) |
| Qwopus3.8-27B-Flash Q6_K | same as Qwen3.8-27B: `-fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2` | 141 code / 115 prose | 128 | 24.7 GiB | [recipe](qwopus3-8-27b-flash-q6-k.md) |

![pair 1 chart](../recipes-pair1.png)

## What held across the models (6 pages, 6 models)

- **`--flash-attn on` changes nothing measurable** on this card and build (auto already picks it). Keep it explicit for hygiene.
- **`--parallel 1` is free and sometimes a gain.** Within 1% on five of six models; +10% at 16k depth and 2.3 GiB saved on Gemma, whose sliding-window layers care about the cache layout. Always set it on a single-user box.
- **q4_0 KV cache is a speed tax at depth, on every model measured: 17%, 23%, 27%, 23%, 29%, 17%.** Decode against a 16k prefix drops by that much on Qwen3.8-27B, Gemma 4 31B, Qwen3.6-35B-A3B, Lightning, Ornith 35B and Qwopus Flash. The 40-item GPQA spot checks stayed within 2 items on five of six; on Ornith q4 KV also dropped the spot check by 5 items (20 to 15), the one move that clears the noise floor, queued for a full 198-item pass.
- **q8_0 KV cache is a memory lever on dense models only.** It saves 0.9 to 2.6 GiB at 32k ctx on the dense 27B/31B for 1 to 5 tok/s; on the three 3B-active MoEs it saves 0.1 to 0.25 GiB (the cache is a sliver of their footprint) and still costs 5 to 7% at depth, so there is nothing to win.
- **MTP pays when the base is slow enough and the head guesses well enough.** Qwen3.8's embedded head (0.61 to 0.99 acceptance) pays 1.8 to 2.3x on a 63 tok/s base; Gemma's community head (0.36 to 0.51) pays 1.5 to 1.7x on 76; Lightning's head accepts 0.59 to 0.87 and is **0.6 to 0.8x** on a 364 tok/s base, because at 2.8 ms per plain token the verification pass costs more than the tokens it saves. Qwopus Flash's embedded head matches Qwen3.8's cell for cell (2.2x code). Rule of thumb from six models: above roughly 300 tok/s, leave the head off.
- **Cold TTFT on 16k tokens: 5.0 to 5.8 s on a dense 27B/31B, 1.6 to 2.2 s on a 3B-active MoE; warm is 0.04 to 0.37 s.** Prompt caching is the flag that matters most and it is on by default (`cache_prompt`).

Queued: Qwen3.8-Flash-Next UD-Q2_K_XL (offload line, its own build); a full-GPQA f16-vs-q4-KV pass on Ornith 35B.
