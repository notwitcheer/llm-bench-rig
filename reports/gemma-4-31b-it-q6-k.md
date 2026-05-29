# Benchmark Report: gemma-4-31B-it (Q6_K)

**Date:** 2026-05-29  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | gemma-4-31B-it |
| Parameters | 30.70 B (dense) |
| Quantization | Q6_K |
| File size | 23.47 GiB |
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
| **MMLU** | **87.82%** | accuracy |
| **ARC-Challenge** | **97.61%** | accuracy |
| **HellaSwag** | **91.95%** | accuracy |
| **HumanEval** | **95.73%** | pass@1 |
| **GSM8K** | **97.50%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 87.52% | 1,319 / 1,507 |
| Humanities | 90.01% | 1,424 / 1,582 |
| Social Sciences | 92.86% | 1,535 / 1,653 |
| Other | 82.80% | 1,878 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 2,486 | 170.3 |
| 512 | 2,932 | 29.7 |
| 2048 | 2,751 | 2.4 |
| 4096 | 2,657 | 1.6 |
| 8192 | 2,520 | 2.6 |
| 16384 | 2,316 | 3.1 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 52.8 | 0.0 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/gemma-4-31b-it-q6-k.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/gemma-4-31b-it-q6-k.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gemma-4-31b-it-q6-k.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gemma-4-31b-it-q6-k.md).*
