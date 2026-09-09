# Recipe: Ornith 1.5 35B-A3B Q4_K_M on one RTX 5090

**The line:**

```
llama-server -m Ornith-1.5-35B-Q4_K_M.gguf --jinja -ngl 99 -c 32768 -fa on -np 1
```

292 tok/s short, 270 decoding against a 16k-token context, 20.8 GiB at 32k ctx, cold 16k TTFT 2.2 s, warm 0.06 s. Keep the KV cache at f16: on this model q4 KV is the one set in the whole recipe series that cost quality as well as speed.

## The six sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 289.2 | 289.2 | 2.18 s | 0.058 s | 267.9 | 21.0 GiB | 20/40 |  |
| + `--flash-attn on` | 289.1 | 289.1 | 2.19 s | 0.059 s | 267.6 | 21.0 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 277.5 | 277.4 | 2.20 s | 0.060 s | 253.3 | 20.8 GiB | 21/40 |  |
| + fa + KV `q4_0/q4_0` | 250.8 | 250.7 | 2.20 s | 0.060 s | 191.2 | 20.6 GiB | 15/40 |  |
| + fa + `--parallel 1` | 292.5 | 292.5 | 2.18 s | 0.059 s | 270.5 | 20.8 GiB |  |  |

(no MTP set: the shipped MTP head is a vLLM artefact and measured negative there in August; no llama.cpp GGUF head exists.)

## Reads

1. **Nothing to tune.** Flash attention and `--parallel 1` land within 1% of base; `-np 1` saves 0.2 GiB. The recipe is the base line plus hygiene, the same as the other 3B-active MoE (Qwen3.6-35B-A3B).
2. **q4 KV: 29% slower at depth and 5 GPQA items down.** 191 vs 268 tok/s against the 16k prefix, and the 40-item spot check fell from 20 to 15 (q8 KV: 21). At n=40 one item is 2.5 points, so 5 items is the first move in six models that clears the noise floor. It is one pass on one model, so it is a flag to run the full 198 items on, not a finding yet; but it is the reason this page says f16 and means it.
3. **KV quantisation saves almost nothing on a 3B-active MoE.** 0.25 GiB for q8, 0.4 GiB for q4, out of 21 GiB. Same as Qwen3.6-35B-A3B and Lightning: the cache is a sliver of the footprint, the dequant tax is the whole story.
4. **The agent numbers.** 2.2 s to prefill 16k tokens cold, 0.06 s warm, decode holds 93% of its short-prompt speed at depth. With GPQA 52 think-off and 82 think-on on the board, this is the reasoning MoE a coding agent on a 24 GB card would run.

## Worth knowing

- 21 GiB at 32k ctx leaves 11 GiB on a 5090: 64k or 128k context is a matter of adding it; on a 24 GB card 32k fits with 3 GiB to spare.
- The 5-item GPQA move on q4 KV is queued for a full 198-item pass (`gpqa` at f16 vs q4 KV, ~1 h think-off) before it goes anywhere near a headline.
- One model, one quant, one build, one pass per cell; the GPQA spot check is 40 items, a screen, not a score.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 10:34 to 11:02 CEST, `recipe-sweep.v2.sh` (spot checks in-run), raw records `results/<slug>/recipe/<set>.jsonl`.
