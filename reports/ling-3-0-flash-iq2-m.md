# Benchmark Report: Ling-3.0-flash-IQ2_M (IQ2_M)

**Date:** 2026-08-09  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Ling-3.0-flash-IQ2_M |
| Parameters | 127.49 B total / 5.1 B active (hybrid MoE, bailingmoe3, 512 experts) |
| Quantization | IQ2_M (bloomer010/Ling-3.0-flash-GGUF, SwiGLU clamp metadata fix 2026-08-07) |
| File size | 39.21 GiB |
| Engine | llama.cpp (CUDA 12.8 (patched)) |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 |
| CPU | AMD Ryzen 5 9600 |
| RAM | 64GB DDR5-5600 |
| OS | Ubuntu 26.04 LTS |
| CUDA | 12.8 (patched) |

**Run context.** Upstream llama.cpp support for bailingmoe3 had not merged at run time; this run uses the open PR [#26608](https://github.com/ggml-org/llama.cpp/pull/26608) branch (head 0266ebca6). All 512 routed experts held in system RAM (`--n-cpu-moe 99`, `-ngl 99`), identical harness and settings to the [Q3_K_M report](ling-3-0-flash-q3-k-m.md). Part of the three-quant ladder measured 2026-08-09; companion piece: [offload-curve sweep](ling-3-offload-sweep.md).

---

## Quality Benchmarks

All benchmarks use generative evaluation via llama-server chat completions. Multiple-choice tasks (MMLU, ARC, HellaSwag) use letter extraction instead of loglikelihood scoring -- results are internally consistent for model comparison but absolute scores may differ from logprob-based evaluations by 5-15%.

### Summary

| Benchmark | Score | Metric |
|-----------|------:|--------|
| **MMLU** | **82.27%** | accuracy |
| **ARC-Challenge** | **95.56%** | accuracy |
| **HellaSwag** | **91.91%** | accuracy |
| **HumanEval** | **89.02%** | pass@1 |
| **GSM8K** | **92.27%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 83.15% | 1,253 / 1,507 |
| Humanities | 79.90% | 1,264 / 1,582 |
| Social Sciences | 89.35% | 1,477 / 1,653 |
| Other | 78.17% | 1,773 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 138.9 | 13.4 |
| 512 | 400.6 | 8.5 |
| 2048 | 414.9 | 1.9 |
| 4096 | 418.1 | 1.7 |
| 8192 | 419.3 | 0.5 |
| 16384 | 420.6 | 0.5 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 45.5 | 0.1 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-iq2-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-iq2-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-iq2-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-iq2-m.md).*
