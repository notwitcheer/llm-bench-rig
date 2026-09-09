# Recipe: Nemotron 3.5 Lightning 30B-A3B Q4_K_M on one RTX 5090

**The line:**

```
llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf --jinja -ngl 99 -c 32768 -fa on -np 1
```

369 tok/s short, 359 decoding against a 16k-token context, 24.1 GiB at 32k ctx, cold 16k TTFT 1.56 s, warm 0.04 s. **Do not load the MTP head**: it makes this model slower. Keep the KV cache at f16.

## The six sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 363.5 | 358.5 | 1.55 s | 0.039 s | 355.2 | 23.7 GiB | 14/40 |  |
| + `--flash-attn on` | 365.6 | 365.4 | 1.55 s | 0.040 s | 356.9 | 23.7 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 356.6 | 356.4 | 1.56 s | 0.040 s | 330.7 | 23.6 GiB | 14/40 |  |
| + fa + KV `q4_0/q4_0` | 348.1 | 347.9 | 1.56 s | 0.040 s | 272.1 | 23.6 GiB | 13/40 |  |
| + fa + `--parallel 1` | 368.7 | 368.6 | 1.56 s | 0.039 s | 358.8 | 23.5 GiB |  |  |
| + fa + MTP `--spec-type draft-mtp --spec-draft-n-max 2` | 297.1 | 225.0 | 1.67 s | 0.053 s | 220.0 | 25.5 GiB |  | 0.87 code / 0.59 prose |

## Reads

1. **The draft head is a loss on a model this fast.** 297 tok/s on code and 225 on prose with MTP against 364 without; 220 against 355 on the 16k prefix. Acceptance is healthy (0.87 code, 0.59 prose), so the head is guessing well: the problem is that at 355 tok/s a plain decode step costs 2.8 ms and the verification pass of two drafted tokens costs more than the two tokens would have. Speculative decoding buys back time on a memory-bound model; a 3B-active hybrid-Mamba model on a 5090 is not memory-bound enough for it to pay. Same verdict as the [day-0 treatment](../nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m.md) in August (MTP n=2 at 0.73 to 0.78x across four workloads, DFlash no better); this run adds the 16k-prefix case, where it is 0.62x.
2. **KV quantisation: nothing to win.** q8 saves 0.1 GiB, q4 saves 0.15 GiB, and they cost 7% and **23% at depth**. The Mamba layers carry no KV cache at all, so the attention layers' cache is a sliver of the footprint. GPQA spot 14, 14, 13 of 40.
3. **The agent numbers.** 1.56 s to prefill 16k tokens cold, 0.04 s warm, decode holds 98% of its short-prompt speed at depth. Fastest cold TTFT of the six models measured so far.

## Worth knowing

- q_avg 83.7 on the board, GPQA 39 think-off: speed is what this model has, not knowledge. The recipe pages do not change that trade, they only make sure the flags do not waste the speed.
- `--spec-draft-n-max 1` was not tested; a single drafted token might break even. Unlikely to beat 364 either way.
- One model, one quant, one build, one pass per cell; the GPQA spot check is 40 items, a screen, not a score.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 10:15 to 10:28 CEST, `recipe-sweep.v2.sh` (spot checks in-run), raw records `results/<slug>/recipe/<set>.jsonl`.
