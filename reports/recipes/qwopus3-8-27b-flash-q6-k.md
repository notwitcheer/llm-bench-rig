# Recipe: Qwopus3.8-27B-Flash Q6_K on one RTX 5090

**The line:**

```
llama-server -m Qwopus3.8-27B-Flash-MTP-Q6_K.gguf --jinja -ngl 99 -c 32768 -fa on -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

141 tok/s on code, 115 on prose, 128 decoding against a 16k-token context, 24.7 GiB at 32k ctx. The MTP head is embedded in the GGUF (the `nextn` tensors) and behaves like the base Qwen3.8-27B's: 0.91 acceptance on code, 0.63 on prose. Keep the KV cache at f16.

## The six sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 63.5 | 63.4 | 5.62 s | 0.120 s | 60.9 | 22.9 GiB | 21/40 |  |
| + `--flash-attn on` | 63.4 | 63.4 | 5.63 s | 0.120 s | 61.0 | 22.9 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 62.6 | 62.5 | 5.66 s | 0.122 s | 59.8 | 22.1 GiB | 20/40 |  |
| + fa + KV `q4_0/q4_0` | 60.3 | 60.2 | 5.66 s | 0.124 s | 50.8 | 21.6 GiB | 20/40 |  |
| + fa + `--parallel 1` | 63.7 | 63.7 | 5.63 s | 0.120 s | 61.2 | 22.5 GiB |  |  |
| + fa + MTP `--spec-type draft-mtp --spec-draft-n-max 2` | 141.2 | 115.1 | 5.90 s | 0.161 s | 128.1 | 24.7 GiB |  | 0.91 code / 0.63 prose |

## Reads

1. **Identical recipe to Qwen3.8-27B Q6_K.** Every cell is within 2% of the base model's page: 63.5 vs 63.1 short, 60.9 vs 60.5 on the prefix, 141 vs 144 with MTP, 0.91 vs 0.93 acceptance on code. The fine-tune changed the weights, not the serving shape, which is what a same-architecture tune should look like on this axis.
2. **MTP is the whole recipe.** 2.2x on code, 1.8x on prose, 2.1x against the 16k prefix, for 1.8 GiB and 40 ms of TTFT.
3. **q4 KV: -17% at depth.** 50.8 vs 60.9 tok/s on the prefix, saving 1.4 GiB; q8 KV costs 1 tok/s and saves 0.85 GiB. GPQA spot 21 / 20 / 20 of 40, inside noise. Same numbers as the base model.
4. **This page is about flags, not quality.** The tune's HumanEval collapse (26.8 vs 94.5, indentation) is on the [treatment report](../qwopus3-8-27b-flash.md); the serving recipe does not change it either way.

## Worth knowing

- Same GPU budget as Qwen3.8-27B Q6_K: 32k context at f16 KV with the MTP head is 24.7 GiB, so a 24 GB card runs this line at `-c 16384`.
- One model, one quant, one build, one pass per cell; the GPQA spot check is 40 items, a screen, not a score.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 10:34 to 11:02 CEST, `recipe-sweep.v2.sh` (spot checks in-run), raw records `results/<slug>/recipe/<set>.jsonl`.
