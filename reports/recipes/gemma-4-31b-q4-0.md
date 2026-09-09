# Recipe: Gemma 4 31B Q4_0 (QAT) on one RTX 5090

**The line, for code:**

```
llama-server -m gemma-4-31B_q4_0-it.gguf --jinja -ngl 99 -c 32768 -fa on -np 1 \
  --model-draft gemma-4-31B-it-MTP-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

130 tok/s on code, 114 on prose, 95 against a 16k prefix, 23.8 GiB. **For chat or long-context work, drop the draft head** and keep `-np 1`: 75.6 tok/s short, 68.7 on a 16k prefix, 20.9 GiB. The community MTP head accepts only 36 to 51% of its drafts on this model, so it is a coding-assistant flag, not a default.

![recipe chart](../recipes-pair1.png)

## The six sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 75.6 | 75.6 | 5.28 s | 0.357 s | 62.3 | 23.2 GiB | 22/40 |  |
| + `--flash-attn on` | 75.6 | 75.6 | 5.28 s | 0.355 s | 62.3 | 23.2 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 70.1 | 70.1 | 5.32 s | 0.217 s | 59.3 | 20.6 GiB | 22/40 |  |
| + fa + KV `q4_0/q4_0` | 61.6 | 61.6 | 5.21 s | 0.144 s | 48.0 | 19.1 GiB | 23/40 |  |
| + fa + `--parallel 1` | 75.6 | 75.6 | 4.97 s | 0.352 s | 68.7 | 20.9 GiB |  |  |
| + fa + MTP `--spec-type draft-mtp --spec-draft-n-max 2` | 129.5 | 114.0 | 5.46 s | 0.368 s | 94.6 | 23.8 GiB |  | 0.49 code / 0.36 prose |

## Reads

1. **`--parallel 1` is the free win here.** Same 75.6 tok/s short, but 68.7 vs 62.3 tok/s against the 16k prefix (+10%), 0.3 s off the cold TTFT, and 2.3 GiB saved. `/props` shows 4 slots at 32k on both, so this is the unified KV cache laid out for one sequence rather than four; on Gemma's interleaved sliding-window attention that layout matters more than it did for Qwen (where `-np 1` moved nothing). Always set it on a single-user box.
2. **The MTP head is workload-dependent.** 1.7x on code, 1.5x on prose, 1.5x on a 16k prefix, from acceptance rates of 0.36 to 0.51, half what Qwen's embedded head reaches. Still a real gain for code; on prose it is 114 vs 76 and worth it if you have the 0.6 GiB, but the head and the base model were quantised by different people and it shows.
3. **KV quantisation costs speed before it costs quality.** q8 KV: 70 vs 75.6 tok/s (-7%), 2.6 GiB saved, GPQA 22/40 = base. q4 KV: 61.6 short, **48.0 on the 16k prefix (-23%)**, 4.1 GiB saved, GPQA 23/40. Same shape as Qwen, larger tax. On a 24 GB card q8 KV is how a 32k context fits; q4 is for when even that does not.
4. **Warm TTFT is 0.36 s, three times Qwen's 0.12 s.** The question tokens are the same; the difference is what Gemma's sliding-window layers have to recompute when a cached prefix is reused. Interesting rather than important, 0.36 s is still instant.

## Worth knowing

- The QAT Q4_0 sits at q_avg 94.3 on the board, tied for first, in 16.4 GB of weights; this recipe is the fastest thing in the top band that a 24 GB card runs with room for context.
- `--spec-draft-n-max 2` was not swept; with acceptance this low a draft of 1 might be as fast and cheaper. A follow-up leg.
- One model, one quant, one build, one pass per cell; the GPQA spot check is 40 items, a screen, not a score.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 08:33 to 09:45 CEST, `recipe-sweep.sh` + `recipe-gpqa-spot.sh`, raw records `results/<slug>/recipe/<set>.jsonl`.
