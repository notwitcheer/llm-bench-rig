# Benchmark Report: Ling-3.0-flash-IQ3_XXS (IQ3_XXS)

**Date:** 2026-08-09  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Ling-3.0-flash-IQ3_XXS |
| Parameters | 127.49 B total / 5.1 B active (hybrid MoE, bailingmoe3, 512 experts) |
| Quantization | IQ3_XXS (bloomer010/Ling-3.0-flash-GGUF, SwiGLU clamp metadata fix 2026-08-07) |
| File size | 47.69 GiB |
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
| **MMLU** | **83.31%** | accuracy |
| **ARC-Challenge** | **96.08%** | accuracy |
| **HellaSwag** | **92.19%** | accuracy |
| **HumanEval** | **93.29%** | pass@1 |
| **GSM8K** | **93.03%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 84.61% | 1,275 / 1,507 |
| Humanities | 81.48% | 1,289 / 1,582 |
| Social Sciences | 90.44% | 1,495 / 1,653 |
| Other | 78.53% | 1,781 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 28.9 | 13.9 |
| 512 | 207.8 | 37.3 |
| 2048 | 338.1 | 15.6 |
| 4096 | 361.3 | 2.9 |
| 8192 | 369.6 | 1.4 |
| 16384 | 374.4 | 1.1 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 39.5 | 0.1 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-iq3-xxs.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-iq3-xxs.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-iq3-xxs.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-iq3-xxs.md).*
