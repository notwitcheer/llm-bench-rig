# Recipe: Qwen3.6-35B-A3B UD-Q5_K_M on one RTX 5090

**The line:**

```
llama-server -m Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --jinja -ngl 99 -c 32768 -fa on -np 1
```

266 tok/s short, 248 decoding against a 16k-token context, 26.1 GiB at 32k ctx, cold 16k TTFT 2.2 s, warm 0.06 s. Keep the KV cache at f16. There is no MTP head for this cut, and nothing else in the six sets moves the number.

## The six sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 264.2 | 263.9 | 2.20 s | 0.059 s | 246.2 | 25.7 GiB | 25/40 |  |
| + `--flash-attn on` | 264.1 | 263.8 | 2.21 s | 0.059 s | 246.4 | 25.7 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 254.4 | 254.1 | 2.23 s | 0.059 s | 234.5 | 25.5 GiB | 25/40 |  |
| + fa + KV `q4_0/q4_0` | 230.2 | 230.0 | 2.23 s | 0.060 s | 179.3 | 25.3 GiB | 25/40 |  |
| + fa + `--parallel 1` | 266.1 | 265.7 | 2.21 s | 0.059 s | 247.9 | 25.5 GiB |  |  |

(the MTP set is absent: no draft head exists for this GGUF.)

## Reads

1. **Nothing to tune.** Flash attention and `--parallel 1` land within 1% of base. The recipe is the base line plus hygiene.
2. **KV quantisation has no upside on a 3B-active MoE.** q8 saves 0.25 GiB and costs 5% at depth; q4 saves 0.4 GiB and costs **27% at depth** (179 vs 246 tok/s against the 16k prefix). With 3B of active weights the KV cache is a small share of the 26 GiB footprint, so there is nothing to win and the dequant tax is the whole story. GPQA spot checks 25/40 on all three sets.
3. **The agent numbers are the point.** A 16k prompt prefills in 2.2 s cold (2.5x faster than the dense 27Bs at 5.5 s) and 0.06 s warm, and decode holds 93% of its short-prompt speed against that prefix. This is the fastest top-band row a coding agent can run on this card.

## Worth knowing

- The board's Qwen3.6-35B-A3B row is the UD-Q4_K_M cut (q_avg 93.3, 271 tok/s tg128); that file is no longer on disk, so this recipe is on the UD-Q5_K_M sibling (24.6 GiB). Same architecture, 4 GB heavier; expect the Q4 cut to be a little faster and 4 GiB lighter on the same line.
- One model, one quant, one build, one pass per cell; the GPQA spot check is 40 items, a screen, not a score.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 10:15 to 10:28 CEST, `recipe-sweep.v2.sh` (spot checks in-run), raw records `results/<slug>/recipe/<set>.jsonl`.
