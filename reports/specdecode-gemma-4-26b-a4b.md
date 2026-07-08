# Three ways to draft, one RTX 5090: MTP vs EAGLE-3 vs DFlash on Gemma 4 26B-A4B — and the consumer-Blackwell tax to run any of them

**Rig:** one RTX 5090 32GB (sm_120) · vLLM 0.21.0 · `nvidia/Gemma-4-26B-A4B-NVFP4` (NVFP4 via MARLIN — consumer Blackwell has no native FP4 kernel, so weights are 4-bit-stored, bf16-compute) · greedy (temp 0) · chat-templated · 5-workload spread, median of 3 · acceptance read from `/metrics`.

Speculative decoding has three live drafter families for Gemma 4, and the field mostly pairs **EAGLE-3 with Gemma**. We ran all three on the same model, same card, same protocol — including the **EAGLE-3 leg nobody publishes** — and measured both axes that matter: realized single-stream speedup, and the acceptance length that explains it.

## The three-way (single-stream, median of 3)

| method | drafter | k | decode tok/s | speedup | accept-len | accept-rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | — | — | 193 | 1.00× | — | — |
| EAGLE-3 | RedHatAI speculator (0.9B) | 3 | 325 | 1.69× | 2.44 | 0.48 |
| MTP | gemma-4 assistant | 4 | 411 | 2.13× | 3.85 | 0.71 |
| DFlash | z-lab (block-16) | 15 | 422 | **2.19×** | 3.45 | 0.16 |

**It's a near-tie at the top: DFlash 2.19× ≈ MTP 2.13×, with EAGLE-3 trailing at 1.69×.** And the *way* they tie is the interesting part. MTP has the higher acceptance — it lands 71% of its 4 drafted tokens (length 3.85). DFlash lands only 16% of its 15 (length 3.45) — yet matches MTP's wall-clock, because it drafts the whole block in **one parallel forward** instead of running a drafter k times. MTP wins on accuracy, DFlash wins on draft-cost; they arrive at the same speed by opposite routes. EAGLE-3's heavier autoregressive draft (k=3, run three times per verify) is the weakest here despite being the "blessed" Gemma pairing — exactly the "drafter overhead eats the gain when the target is small/active-light" prediction for a 4B-active MoE. (Every method obeys speedup < accept-len, as it must — acceptance length is the ceiling.)

## DFlash is feast-or-famine; MTP is the steady all-rounder

The average hides the real story. Speedup by workload, least- to most-predictable:

| | prose | Q&A | code | JSON | repetitive |
|---|---:|---:|---:|---:|---:|
| EAGLE-3 | 1.30× | 1.50× | 1.75× | 2.01× | 1.88× |
| MTP | 1.58× | 1.89× | 2.01× | 2.50× | 2.69× |
| DFlash | **1.04×** | 1.32× | 1.55× | 2.68× | **4.37×** |

DFlash is near-useless on prose (1.04×) and **crushing on structured/repetitive text (4.37×)** — its block draft only pays when the next 16 tokens are guessable. MTP rises gently and steadily. **Pick by workload:** DFlash for codegen/structured/JSON output, MTP if your traffic is mixed or prose-heavy.

## A rigor note: we caught our own measurement bug

The first pass said "DFlash 3.9×, runaway winner." It was wrong, twice. A single-run prose outlier inflated DFlash's average; and the acceptance numbers were being corrupted by a parser bug — vLLM emits `num_accepted_tokens_total` next to a per-position counter series (`..._per_pos_total{position=...}`, a labelled Counter — corrected 2026-07-08, we previously called it a histogram) and a `..._created` timestamp (standard OpenMetrics convention), and our reader aliased them together, so Python's non-deterministic set ordering intermittently read **0.0** instead of the true count. Fixed (exact-match the `_total` counters), regression-tested, and re-ran median-of-3 — which is where the honest near-tie above comes from. The tell was physical: the buggy run reported speedup > acceptance length, which is impossible. We don't ship numbers that violate their own ceiling.

## The consumer-Blackwell tax (why this is a rig story, not a benchmark)

None of this ran out of the box. sm_120 + a multimodal MoE + bleeding-edge drafters took six fixes, each a real wall:

1. **FlashInfer** JIT-fails on sm_120 ("no supported CUDA arch for major version 12") → `VLLM_USE_FLASHINFER_SAMPLER=0` (we're greedy anyway).
2. **Multimodal MM-budget** validation aborts unless `--max-num-batched-tokens ≥ 2496`.
3. **NVFP4 has no native FP4 kernel** on consumer Blackwell → `VLLM_NVFP4_GEMM_BACKEND=marlin` (memory win, not FLOPS — bf16 compute).
4. **Instruction-tuned model degenerates on raw `/completions`** ("is is is is") → must use chat-templated requests, or acceptance is measured on garbage.
5. **DFlash + Gemma's multimodal attention** is a backend deadlock (vLLM #42068): the target forces TRITON_ATTN, DFlash needs non-causal, flash_attn rejects "partial multimodal full attention" — **only `FLEX_ATTENTION` (arbitrary-mask) satisfies both.**
6. **KV fit** on 32GB is razor-thin; DFlash's k=15 verify buffers are the tightest of the three.

## A note on the dense 31B (excluded)

We tried the dense Gemma-4-31B too. nvidia's official NVFP4 31B is **partially quantized** (attention + vision kept bf16) → 31GB, won't fit a 32GB card with a drafter. The only fitting option was a *community* NVFP4 quant — which shifted the target distribution enough to collapse every drafter's acceptance and produced internally inconsistent numbers (speedup > acceptance length). So the honest result is: **the dense 31B doesn't cleanly fit a 32GB card for speculative decoding** — and we don't publish a number we can't stand behind.

## Verdict

On a single 5090, single stream: **MTP and DFlash both land ~2.2×, and the right pick is the workload.** Structured/repetitive output (code, JSON, logs) → DFlash, up to ~4×. Mixed or prose-heavy → MTP, steady ~2× and it ships with the model (no extra drafter). EAGLE-3 is the weakest of the three here despite being Gemma's default speculator. Worth it for a local single-user agent; we make no claim about batched serving — the concurrency sweep wasn't reproducible enough to stand behind.

## Reproduce

`scripts/run_specdecode_leg.sh <target_slug> <model> <leg>` (one vLLM server per leg; FLEX_ATTENTION + sm_120 env baked in; `MAXLEN`/`GPUUTIL`/`REPS` overridable) · `scripts/bench_specdecode.py` (chat-templated workload spread + `/metrics` acceptance, `--reps` median) · `scripts/aggregate_specdecode.py` · chart: `scripts/chart_specdecode.py`. All on vLLM 0.21.0, one RTX 5090.
