# Muse Glimmer 30B — UD-Q4_K_XL on one RTX 5090 (day-0 treatment)

**Date:** 2026-08-11 (benched 2026-08-10, release day) · **Hardware:** RTX 5090 32GB, 64GB DDR5-5600 · **Engine:** llama.cpp master b10349 (`62bf73d25`, the Muse Glimmer merge commit — PR [#26841](https://github.com/ggml-org/llama.cpp/pull/26841), merged the morning of release)

## Model

| | |
|---|---|
| Model | Muse-Glimmer-30B (Meta) — dense VLM: 28B text decoder + 2B Perception Encoder vision tower |
| Architecture | hybrid attention: (SWA-2048 ×3 + full-attention NoPE) ×13 = 52 layers; 16:1 GQA (32 Q heads / 2 KV heads); Q-K norm with query scaling; 128K context |
| Quantization | UD-Q4_K_XL (unsloth), 15.9GB single file; source repo `unsloth/Muse-Glimmer-30B-GGUF` |
| License | Apache 2.0 |

Run context: model fully resident in VRAM (no offload), `--jinja`, temperature 0, think classification **ON** (see below). Text benches exercise the text decoder only; the 2B vision tower rides along in the file but is not loaded for these tasks.

## Speed (llama-bench)

| Metric | Value |
|---|---:|
| tg128 | **85.72 tok/s** (±0.07) |
| pp512 | 4,769 tok/s |
| pp2048 | 4,680 tok/s |
| pp8192 | 4,539 tok/s |
| pp16384 | 4,429 tok/s |
| VRAM peak | 15,726 MiB |

Under chat-server conditions (template + sampling overhead, 256-token generations) the same setup delivers ~83.4 tok/s — the honest baseline for the speculative-decoding numbers below.

For contrast: the offloaded 120B-class MoEs on this card (gpt-oss-120B MXFP4, Ling-3.0-flash Q3_K_M via `--n-cpu-moe`) generate at 46–47 tok/s because each token waits on system RAM. A dense 30B that fits runs at nearly twice that with half the card spare.

## Quality Benchmarks (Thinking ON)

| Task | Score | Items |
|---|---:|---:|
| MMLU | 80.6 | 7,010 (50% sample) |
| ARC-Challenge | 93.5 | 1,172 |
| HellaSwag | 88.8 | 5,021 (50% sample) |
| GSM8K | 96.7 | 1,319 |
| HumanEval | 95.7 pass@1 | 164 |
| **q_avg** | **91.1** | |

### The always-on reasoning channel

Muse Glimmer's chat template exposes `reasoning_strength` (default `high`) instead of the usual `enable_thinking` toggle — and there is no off value. Probed directly against the template: `low`, `minimal`, `none`, and `off` all still produce ~60 reasoning tokens before a one-letter MC answer. The harness's completion-length gate (armed for think-off runs, threshold 50 tokens) correctly aborted the first suite attempt; the model was then reclassified to the Thinking-ON board and run at `reasoning_strength: low`. If you benchmark this model, verify your think-off switch actually changed anything — it probably didn't.

## DFlash speculative decoding (shipped drafter)

Muse Glimmer ships with a DFlash draft model (`dflash-kquant.gguf`, 1.6GB). Measured on four workloads (8 prompts × 256 tokens each, temperature 0) against the 83.4 tok/s chat-server baseline:

| Workload | Base tok/s | DFlash tok/s | Speedup |
|---|---:|---:|---:|
| code | 83.4 | 119.0 | **1.43x** |
| repetitive | 83.3 | 124.0 | **1.49x** |
| chat | 83.3 | 89.9 | 1.08x |
| prose | 83.5 | 79.9 | **0.96x** (slowdown) |

Draft acceptance on prose ran 44–61% (mean accepted length ~2.4) — below break-even for the drafting overhead. Speculative decoding remains workload-shaped: large wins on structured output, a net loss where acceptance drops.

**Flag gotcha (day-0 llama.cpp):** `-md <dflash.gguf>` alone loads the draft model and silently never engages it — all workloads return exactly 1.00x. The DFlash path must be selected explicitly: `--spec-type draft-dflash`. (The build's `[spec] failed to measure draft model memory` warning at startup is cosmetic and appears in both working and non-working configurations.)

## Reads

1. Meta's return to open weights is a strong local daily driver: q_avg 91.1 within a point of gpt-oss-120B (91.9) from a file a quarter the size, at nearly double the generation speed, with vision capability the text board doesn't credit.
2. The 16:1 GQA and hybrid SWA/NoPE attention keep both the KV cache and long-context prefill healthy: pp16384 holds 93% of pp512 throughput.
3. The always-on reasoning channel is a real deployment consideration: every request pays a reasoning-token latency tax, and harnesses that assume think-off MC behaviour will misfire.
4. The shipped DFlash drafter is worth enabling for code/structured workloads (1.4–1.5x) and worth disabling for prose serving.
