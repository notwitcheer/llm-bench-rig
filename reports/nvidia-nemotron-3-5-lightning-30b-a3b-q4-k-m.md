# Benchmark Report: NVIDIA-Nemotron-3.5-Lightning-30B-A3B (Q4_K_M)

**Date:** 2026-08-12  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | NVIDIA-Nemotron-3.5-Lightning-30B-A3B |
| Parameters | 31.58 B total / 3 B active (nemotron_h hybrid MoE) |
| Quantization | Q4_K_M (source: lmstudio-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) |
| File size | 22.83 GiB |
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

**Run context.** GGUF leg: llama.cpp master worktree at build b10371 (5d16e81dd), built with the rig's pinned toolchain (gcc-14, nvcc 12.8), all layers resident (`-ngl 99`), no expert offload. Model released 2026-08-11; benched the same day. The NVFP4 leg below ran on the identical harness the following hours, against the official `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` build via vLLM 0.25.1 on the native cutlass sm_120 FP4 path.

## Quality Benchmarks

All benchmarks use generative evaluation via llama-server chat completions. Multiple-choice tasks (MMLU, ARC, HellaSwag) use letter extraction instead of loglikelihood scoring -- results are internally consistent for model comparison but absolute scores may differ from logprob-based evaluations by 5-15%.

### Summary

| Benchmark | Score | Metric |
|-----------|------:|--------|
| **MMLU** | **77.93%** | accuracy |
| **ARC-Challenge** | **92.15%** | accuracy |
| **HellaSwag** | **80.64%** | accuracy |
| **HumanEval** | **82.32%** | pass@1 |
| **GSM8K** | **85.60%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 74.78% | 1,127 / 1,507 |
| Humanities | 78.19% | 1,237 / 1,582 |
| Social Sciences | 86.45% | 1,429 / 1,653 |
| Other | 73.63% | 1,670 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 4,559 | 24.3 |
| 512 | 11,698 | 39.7 |
| 2048 | 11,693 | 69.3 |
| 4096 | 11,571 | 37.6 |
| 8192 | 11,440 | 13.3 |
| 16384 | 11,198 | 21.7 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 377.4 | 1.7 |

---

## NVFP4 second data point (official build, vLLM 0.25.1)

Same model, same card, same five-task harness (50% sample, seed 42, think-off): the official NVFP4 release (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, 21.6 GB weights) served through vLLM 0.25.1 on the native cutlass sm_120 FP4 path, versus this Q4_K_M GGUF through llama.cpp.

| | Q4_K_M (llama.cpp) | NVFP4 (vLLM 0.25.1) |
|---|---:|---:|
| MMLU | 77.93 | 77.72 |
| ARC-Challenge | 92.15 | 92.58 |
| HellaSwag | 80.64 | 81.40 |
| GSM8K | 85.60 | 86.96 |
| HumanEval | 82.32 | 81.71 |
| **q_avg** | **83.7** | **84.1** |
| decode (see note) | 377.4 tok/s (tg128) | 410.5 tok/s chat-server median (462.2 at 2k-token prompt) |
| weights on disk | 22.83 GiB | 21.6 GB |

Speed convention caveat: the two decode figures are **not** the same measurement — llama-bench tg128 vs chat-server completion throughput (completion tokens / (total − TTFT), 3-run medians). They support "same speed class, NVFP4 ahead on the native path", not a precise multiplier. Quality numbers ARE directly comparable (identical harness both legs).

Run integrity note: the NVFP4 quality pass crashed mid-GSM8K once with `CUDA error: misaligned address` (vLLM engine death); the three MC tasks completed before the crash and GSM8K + HumanEval were re-run clean on a fresh server with the identical recipe. Recorded as-is in `quality_nvfp4.json`.

Two sm_120 traps for reproducers: vLLM 0.25.1's flashinfer JIT fails with `nvcc fatal: Unsupported gpu architecture 'compute_120f'` unless `CUDA_HOME` points at CUDA 13 (the 12.8 nvcc doesn't know `120f`); and the misaligned-address engine death happened once across the whole board run and did not recur on the rerun — treat it as an intermittent hazard on this stack, not a reproducible bug we can pin.

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m.md).*
