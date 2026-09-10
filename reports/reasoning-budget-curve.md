# How much thinking is worth paying for: GPQA-diamond think-on at three completion budgets on one RTX 5090

**TL;DR.** Two think-on models, the same 198 GPQA-diamond items, the same server, and only the completion budget (`max_tokens`) moved: 4,096, 8,192, 16,384. Qwen3.8-27B Q6_K scores **60.1 / 70.2 / 79.3**, Ornith 1.5 35B-A3B Q4_K_M **70.2 / 77.3 / 81.8**. Every point lost at a smaller budget is an item that ran into the cap: in all four new legs, 100% of the items that were correct at 16k and wrong at 4k or 8k are items the budget truncated (Qwen 42/42 and 23/23, Ornith 25/25 and 11/11), and no uncapped item got worse. Items that finished inside the budget were correct 94.7 to 96.1% of the time at every budget, for both models. The median item thinks for roughly 950 tokens, but 30 to 43% of the set runs past 4k or 8k, and that tail is where the hard questions live. Cost: 4k to 8k buys Qwen +10.1 points for +66% completion tokens (3,972 to 5,636 tokens per correct answer) and Ornith +7.1 points for +46% (3,248 to 4,733); 8k to 16k buys Qwen +9.1 points for a further +55% (7,740 per correct) and Ornith +4.5 points for +57% (6,996). Even at 16k, 22% of items still hit the cap on both models. Ornith is the cheaper thinker at every budget, by 700 to 900 tokens per correct answer. The 16k legs also reproduced the August 16k runs item for item (0 of 198 differ), the deterministic-decode check for free.

![reasoning budget curve](reasoning-budget-curve.png)

## Why this study

The local-inference frustrations harvest of 2026-09-08 (222 replies to one X thread) kept returning to reasoning models on consumer cards: "27B blows the budget on reasoning", doom loops, models that "don't finish their tasks". The published fix is usually "raise max_tokens", which is a memory and latency cost with no number attached. This report attaches the number: for two models this box runs, what does each doubling of the completion budget buy in accuracy, and what does it cost in tokens per correct answer? Tokens per correct is now a standing output of the think-on GPQA eval in this repo (`lib/evals/gpqa.py`, since ee3e1c8), so every future think-on leg carries it.

## Setup

- **Hardware and server:** RTX 5090 32 GB, llama.cpp llama-server started through `lib.quality.start_llama_server` (the harness's standard line), `ctx_size 24576`, the resident server drained for the duration and restored after. Models fully resident.
- **Models:** `Qwen3.8-27B-Q6_K.gguf` (dense, thinking on); `Ornith-1.5-35B-Q4_K_M.gguf` (35B-A3B MoE, thinking on). Both were already on the board with think-on 16k legs from August (`results/<slug>-thinkon/gpqa.json`).
- **Eval:** GPQA-diamond, 198 items, zero-shot, deterministic per-item option shuffle (seed `42-<idx>`), temperature 0, `think=True`, letter extraction with the reasoning-block fallback used by every think-on leg in this repo. The only variable across legs is `max_tokens` (4,096 / 8,192 / 16,384). `capped` means the completion reached `max_tokens`.
- **Runs:** 4k and 8k legs 2026-09-09 15:25 to 22:16 CEST in one script (`budget-curve.sh`, night-lib rails, exact-slug artefacts `results/<slug>-thinkon/budget-<N>/`), Qwen then Ornith. 16k token legs 2026-09-10 07:07 to 13:48 CEST (same script, budget 16384, Ornith 1 h 9 min, Qwen 5 h 32 min; the August 16k legs recorded accuracy only). Per-item sidecar `gpqa_tokens.jsonl` (`idx, completion_tokens, correct, capped`).
- **Noise band:** one item is 0.5 points; gaps under about 3 points between two legs are within the 198-item band. The differences reported here are 4.5 to 19 points; the 8k to 16k step on Ornith (+4.5) is the only one near the band.

## Results

| model | budget | accuracy | capped items | unparsed | median completion tokens | tokens per correct | completion tokens, total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q6_K | 4,096 | **60.1** (119/198) | 85 (43%) | 46 | 2,274 | 3,972 | 472,692 |
| Qwen3.8-27B Q6_K | 8,192 | **70.2** (139/198) | 68 (34%) | 31 | 2,274 | 5,636 | 783,365 |
| Qwen3.8-27B Q6_K | 16,384 | **79.3** (157/198) | 43 (22%) | 14 | 2,274 | 7,740 | 1,215,233 |
| Ornith 1.5 35B-A3B Q4_K_M | 4,096 | **70.2** (139/198) | 75 (38%) | 30 | 2,096 | 3,248 | 451,533 |
| Ornith 1.5 35B-A3B Q4_K_M | 8,192 | **77.3** (153/198) | 60 (30%) | 19 | 2,096 | 4,732 | 724,068 |
| Ornith 1.5 35B-A3B Q4_K_M | 16,384 | **81.8** (162/198) | 43 (22%) | 14 | 2,096 | 6,996 | 1,133,288 |

The median completion length is identical across all three legs of the same model (2,274 tokens for Qwen, 2,096 for Ornith) because greedy decoding produces the same text until the cap; only the tail changes. The 16k legs returned the same 198 correct/incorrect flags as the August 16k runs, so greedy think-on on this build is reproducible run to run.

### Where the points go: capped vs finished, and the per-item diff against 16k

| model | budget | finished items | correct among finished | capped items | correct among capped | correct at 16k, wrong here | ...of which capped here | wrong at 16k, correct here |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q6_K | 4,096 | 113 | 107 (94.7%) | 85 | 12 (14.1%) | 42 | 42 | 4 |
| Qwen3.8-27B Q6_K | 8,192 | 130 | 124 (95.4%) | 68 | 15 (22.1%) | 23 | 23 | 5 |
| Qwen3.8-27B Q6_K | 16,384 | 155 | 149 (96.1%) | 43 | 8 (18.6%) | 0 (same run repeated) | 0 | 0 |
| Ornith 1.5 35B-A3B Q4_K_M | 4,096 | 123 | 117 (95.1%) | 75 | 22 (29.3%) | 25 | 25 | 2 |
| Ornith 1.5 35B-A3B Q4_K_M | 8,192 | 138 | 131 (94.9%) | 60 | 22 (36.7%) | 11 | 11 | 2 |
| Ornith 1.5 35B-A3B Q4_K_M | 16,384 | 155 | 148 (95.5%) | 43 | 14 (32.6%) | 0 (same run repeated) | 0 | 0 |

The 8k-to-16k diff (rows 2 and 5 against rows 3 and 6) has the same shape: Qwen lost 23 items at 8k, all 23 capped, Ornith 11 of 11. The "correct among capped" column is not zero because the answer letter sometimes appears in the reasoning before the cap and the fallback extractor picks it up; it is 14 to 37%, close to what guessing plus a partial chain would give, against ~95% for finished items.

## Reads

1. **Accuracy is a function of how many items the budget cuts off, nothing else.** Across the four smaller-budget legs, every item lost against the 16k reference was capped (101 of 101), and finished items scored 94.7 to 96.1% regardless of budget or model. A smaller budget does not make the model worse at the items it can finish; it removes the items it cannot.
2. **The tail is fat and the tail is the hard part.** At 16k, finished items have a median of ~1,240 tokens and a 75th percentile of 3.2k (Ornith) to 5.0k (Qwen), yet 43% (Qwen) and 38% (Ornith) of the set runs past 4,096 tokens, 34% / 30% past 8,192, and 22% of both still past 16,384. The longest finished chains were 15.1k and 15.6k tokens. The questions that need the room are disproportionately the ones the model gets right when given it.
3. **The price of a point rises with the budget.** 4k to 8k: Qwen pays +66% completion tokens for +10.1 points, Ornith +46% for +7.1. 8k to 16k: Qwen +55% for +9.1 points, Ornith +57% for +4.5. Total tokens across the set nearly triple from 4k to 16k (Qwen 473k to 1.22M) for +19 points. Ornith's 8k to 16k step is the first one inside the noise band's neighbourhood, and its 16k capped share (22%) equals Qwen's, so the next doubling would likely buy it less than Qwen.
4. **Ornith thinks cheaper.** At every budget it caps fewer items (38 vs 43%, 30 vs 34%) and spends 700 to 900 fewer tokens per correct answer, and its 3B-active MoE decodes 4.6 times faster than the dense 27B on this card without a draft head (289 vs 63 tok/s on the same server line, `reports/recipes/`), so the wall-clock gap per correct answer is far larger than the token gap. This is a comparison of two quants of two different models, not a quant ladder; it says what a buyer of either gets, not why.
5. **For an agent harness, the actionable setting is the cap, not the model.** With 30% of items still capped at 8k and 22% at 16k on both models, `max_tokens` below 16k on a reasoning model turns a 79 to 82% model into a 60 to 77% one on exactly the questions that motivated reasoning in the first place. If the memory for a 16k+ completion window is not there (the f16 KV cache at 32k context on Qwen3.8-27B is about 1.8 GiB on this build, from the recipe sweep's q8 saving of 0.9 GiB), a smaller model with the full budget is the better trade than a bigger model with a truncated one.

## Honest limits

- Two models, one quant each, one hardware/server stack. The shape (losses = capped items) is likely general for greedy think-on evaluation; the slopes are these two models' own.
- Greedy decoding. The 198-item band is stated above; the 16k legs of 2026-09-10 repeated the August 16k runs and matched them on all 198 items, so run-to-run noise on this build is zero for greedy think-on and the band applies to item sampling, not to reruns.
- The curve stops at 16k. With 22% of items still capped there, a 32k leg would move both scores again; it is a ~10 h job for the dense model and is not scheduled.
- GPQA-diamond is a multiple-choice science set; "tokens per correct" on an agentic task would need the agentic evals in this repo, which do not run think-on yet.

## Files

- `results/qwen3-8-27b-q6-k-thinkon/budget-{4096,8192,16384}/` and `results/ornith-35b-thinkon/budget-{4096,8192,16384}/`: `gpqa.json`, `gpqa_progress.json`, `gpqa_tokens.jsonl`.
- `results/<slug>-thinkon/gpqa.json`: the August 16k accuracy legs.
- Chart script: `scripts/chart_budget_curve.py` (reads the mirror of the results tree).
