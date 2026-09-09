# Recipe: Qwen3.8-Flash-Next UD-Q2_K_XL on one RTX 5090 (expert offload)

**The line:**

```
llama-server -m Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf --jinja -ngl 99 --n-cpu-moe 22 \
  -c 32768 -fa on -np 1 --cache-type-k q8_0 --cache-type-v q8_0
```

52 to 54 tok/s decoding against a 16k-token context, cold 16k TTFT **29.4 s**, warm 0.32 s, 29.3 GiB at 32k ctx with q8 KV (29.7 at f16). The only model in the series where q8 KV earns its place: -1% decode for 0.5 GiB, and at 29.7 GiB on a 31.5 GiB card that 0.5 GiB is the margin. 59 GB of system RAM holds the offloaded experts; see the [treatment report](../qwen3-8-flash-next-ud-q2-k-xl.md) for the `--n-cpu-moe` ladder that picked 22.

**If you code and can live at 24k context:** add `--spec-type draft-mtp --model-draft mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-draft-n-max 2` on the [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) build, at `-c 24576`: 1.41x on code, 1.15x against the 16k prefix, 31.3 GiB peak, zero headroom.

## The seven sets

| flag set | code tok/s | prose tok/s | 16k cold TTFT | 16k warm TTFT | tok/s on 16k prefix | VRAM peak | GPQA 40-item | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (`--jinja -ngl 99 -c 32768`) | 50.9 | 47.0 | 29.39 s | 0.316 s | 52.9 | 29.7 GiB | 23/40 |  |
| + `--flash-attn on` | 59.2 | 59.0 | 29.11 s | 0.316 s | 53.7 | 29.7 GiB |  |  |
| + fa + KV `q8_0/q8_0` | 58.3 | 52.6 | 29.36 s | 0.317 s | 52.2 | 29.3 GiB | 23/40 |  |
| + fa + KV `q4_0/q4_0` | 51.0 | 50.2 | 29.46 s | 0.317 s | 45.9 | 29.0 GiB | 27/40 |  |
| + fa + `--parallel 1` | 52.7 | 50.5 | 29.36 s | 0.316 s | 53.7 | 29.4 GiB |  |  |
| + fa + MTP `--spec-type draft-mtp --spec-draft-n-max 2` | 72.8 | 48.5 | 29.91 s | 0.338 s | 60.7 | 31.3 GiB |  | 0.93 code / 0.62 prose |
| g_fa_np1_24k | 51.7 | 46.8 | 29.41 s | 0.317 s | 52.8 | 29.2 GiB |  |  |

Sets a to e at ctx 32768; the MTP head does not fit at 32k (CUDA out of memory loading the draft, `f_fa_mtp.server-failed.log`), so set f ran at ctx 24576 with set g as its same-context base.

## Reads

1. **Read the long-context column, not the short one.** Short-prompt decode swings 47 to 59 tok/s across five sets that should agree, while decode against the 16k prefix sits at 52 to 54 on every one of them. Under expert offload the CPU side jitters run to run; the long lane, dominated by steady expert streaming, does not. Every claim on this page uses the long-context number.
2. **q4 KV: -13% at depth, the mildest tax of the seven models.** 45.9 vs 52.9 tok/s. Decode here is bound by the CPU expert path, not GPU attention, so dequantising the cache costs proportionally less. Still a loss for 0.7 GiB. GPQA spot 27 / 23 / 23 of 40 for q4 / q8 / f16 is inside the 40-item noise band (see the Ornith page).
3. **MTP pays 1.4x on code, not the 2.2x the resident Qwen3.8-27B gets, at the same 0.93 acceptance.** Each verification step still streams the CPU-resident experts; the draft head saves GPU attention passes, which are not the bottleneck under offload. Prose at 0.62 acceptance is a wash (1.04x). Cost: 2.2 GiB and 8k of context.
4. **Cold TTFT on 16k tokens is 29.4 s, and the derived estimate said 19.8.** `16384 / pp512@d8192` is within 4% on resident models and 1.5x optimistic here: prefill through CPU-resident experts slows with depth faster than one 8k probe shows. The depth table carries the measured number; the field guide entry has a dated addendum. Warm (prefix cached) it is 0.32 s, so for an agent loop the cold cost is paid once per session.

## Worth knowing

- 24 GB card: the ladder's n=28 row (52.5 tok/s short, 23.0 GiB) leaves no room for the MTP head; the base line with q8 KV is the recipe there.
- The page-cache edge from the treatment report applies: at n=28 on a 59 GB box the RAM-side working set thrashes; at n=22 it does not.
- One model, one quant, two builds (master for a to e and g, PR #28243 for f), one pass per cell.

## Method

One `llama-server` per flag set (llama.cpp b10371, the build the board's resident rows use), `--jinja -ngl 99 --ctx-size 32768 --host 127.0.0.1`, plus the set's flags. Measured with `scripts/recipe_bench.py`, the rig's served lane: the four short workloads (prose, code, repetitive, chat; 8 prompts each, 256 tokens, temperature 0, one discarded warm-up) and a **long lane** with a synthetic repository dump as system prompt, sized to exactly 16,384 tokens with the server's `/tokenize`, followed by 8 short coding questions. The long lane runs twice: **cold** (`cache_prompt: false`, the whole prompt is prefilled on every request, the first turn of an agent session) and **warm** (`cache_prompt: true` after one primer, only the question is prefilled, every later turn). p50 over 8 requests per cell; TTFT is the time to the first content chunk as the client sees it, tok/s is `completion_tokens / (total_s - ttft_s)`. VRAM peak sampled at 0.5 Hz across serve + bench. The 40-item GPQA-diamond spot check (think-off, greedy, the standing harness) runs on the base set and on both KV-quantised sets, in a separate pass with identical flags; at n=40 one item is 2.5 points, so only moves of 3 or more items are read as signal. Server `/props` confirmed 4 slots at n_ctx 32768 on every set (unified KV, so `--parallel 1` changes the cache layout, not the per-slot window). Run 2026-09-09 12:35 to 13:16 CEST (sets a to e, ctx 32768) and 14:48 to 15:04 CEST (sets g and f, ctx 24576), `recipe-sweep.v2.sh` (spot checks in-run), raw records `results/<slug>/recipe/<set>.jsonl`.
