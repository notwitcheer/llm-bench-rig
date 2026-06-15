# Benchmark Report: north-mini-code (unknown)

**Date:** 2026-06-15  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Findings

Cohere **North-Mini-Code-1.0** — a 30B-total / **3B-active** MoE (`cohere2moe`), Apache-2.0, marketed for
agentic coding. The arch merged into llama.cpp on 2026-06-13 ([PR #24260](https://github.com/ggml-org/llama.cpp/pull/24260),
validated on an RTX 5090 inside the PR), so these are the first consumer-Blackwell numbers. Run at **Q6_K**
(23.8 GiB, peak 25.6/32 GB VRAM, no offload), think-OFF to match the board.

**1. It lands LAST on the 9-model agentic board.** Agentic Score **90.4** (task-success 83.3%, tool-eff 0.94),
tied with Nex-N2-mini and below generalist A3B models (Qwen3.5-35B-A3B 97.5, Nemotron-Cascade-2-30B-A3B 96.9).
Tool-efficiency is fine; task-success is the gap. The agentic-coding tuning doesn't translate to top
tool-driving on this harness. *(chart: `north-mini-board.png`)*

**2. ARC-Challenge is reasoning-gated — and the think-OFF board understates a reasoning model.**
ARC-Challenge **60.2% think-OFF → ~95% think-ON** (+35pt; think-ON is a 40-question spot-check, 38/40).
The Challenge subset is selected for multi-step reasoning; forcing a bare-letter answer (think-OFF) measures
the wrong thing. GSM8K is unaffected (95.8% either way) because it's generative — the model writes its
solution regardless. North-Mini is genuinely a reasoning model (`<|START_THINKING|>` tokens; its template
gates on `reasoning`/`reasoning_effort`, not `enable_thinking`). *(chart: `north-mini-arc-gate.png`)*

**3. The reality anchor can't adjudicate it (different protocol).** Cohere self-reports **67.6% SWE-bench
Verified** (full set, their scaffold, 1×H100 FP8) — which would top this board's best *measured* resolve (63%).
But the rig's anchor axis is our native loop on a 30-hardest-bug subset, graded officially. Different scaffold,
subset, and difficulty: the two numbers don't share an axis. So North-Mini is **last on synthetic, claims-best
on real, and the gap is unresolved** — either the synthetic agentic bench under-rates a model trained on real
SWE harnesses, or the vendor number doesn't reproduce on a standardized loop (the Qwopus-Coder precedent:
claimed 67%, scored 57% on our subset). A fresh same-protocol run would resolve it; out of scope here.

**Coder strengths are real:** HumanEval **86.6%**, GSM8K **95.8%**, decode **305 tok/s** (Q6_K). The vendor
"2.8x output throughput vs Devstral Small 2" is a 1×H100 FP8 number; on a 5090 you simply get fast A3B decode.

> **Methodology note:** benchmarking this reasoning model think-OFF first produced a contaminated, ~26×-slower
> run — the harness's `enable_thinking:False` is a no-op for `cohere2moe`, which gates on `reasoning`. Fixed in
> the harness (send `reasoning`/`reasoning_effort` + serve with `--jinja`); verified 106→4 tokens/question.

---

## Model

| Field | Value |
|-------|-------|
| Model | north-mini-code |
| Parameters | 30.48 B (dense) |
| Quantization | unknown |
| File size | 23.76 GiB |
| Engine | llama.cpp (CUDA 12.8 (patched)) |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 |
| CPU | AMD Ryzen 5 9600 |
| RAM | 64GB DDR5-5600 |
| OS | Ubuntu 26.04 LTS |
| CUDA | 12.8 (patched) |

---

## Quality Benchmarks

All benchmarks use generative evaluation via llama-server chat completions. Multiple-choice tasks (MMLU, ARC, HellaSwag) use letter extraction instead of loglikelihood scoring -- results are internally consistent for model comparison but absolute scores may differ from logprob-based evaluations by 5-15%.

### Summary

| Benchmark | Score | Metric |
|-----------|------:|--------|
| **MMLU** | **73.32%** | accuracy |
| **ARC-Challenge** | **60.24%** | accuracy |
| **HellaSwag** | **70.82%** | accuracy |
| **HumanEval** | **86.59%** | pass@1 |
| **GSM8K** | **95.83%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 66.49% | 1,002 / 1,507 |
| Humanities | 72.25% | 1,143 / 1,582 |
| Social Sciences | 85.12% | 1,407 / 1,653 |
| Other | 70.02% | 1,588 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 3,661 | 20.2 |
| 512 | 9,468 | 59.1 |
| 2048 | 9,302 | 48.0 |
| 4096 | 9,026 | 17.3 |
| 8192 | 8,937 | 71.5 |
| 16384 | 8,643 | 98.6 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 304.8 | 1.2 |

---

## Methodology

### Evaluation Framework

Custom generative evaluators built for this rig. All benchmarks run through llama-server's `/v1/chat/completions` endpoint.

- **Scoring:** Generative evaluation (not loglikelihood)
- **Thinking:** disabled
- **MCQ scoring:** First valid letter extracted from response (A/B/C/D)
- **Sampling:** 50% of dataset used
- **Temperature:** 0 (deterministic)
- **Max tokens:** 2,048
- **GPU offload:** All layers (`-ngl 99`)

---

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/north-mini-code.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/north-mini-code.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/north-mini-code.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/north-mini-code.md).*
