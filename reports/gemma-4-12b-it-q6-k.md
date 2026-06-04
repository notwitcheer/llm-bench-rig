# Benchmark Report: gemma-4-12b-it (Q6_K)

**Date:** 2026-06-04  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | gemma-4-12b-it |
| Parameters | 11.91 B (dense) |
| Quantization | Q6_K |
| File size | 9.11 GiB |
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
| **MMLU** | **78.86%** | accuracy |
| **ARC-Challenge** | **94.03%** | accuracy |
| **HellaSwag** | **81.62%** | accuracy |
| **HumanEval** | **87.20%** | pass@1 |
| **GSM8K** | **96.36%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 77.84% | 1,173 / 1,507 |
| Humanities | 77.81% | 1,231 / 1,582 |
| Social Sciences | 88.57% | 1,464 / 1,653 |
| Other | 73.19% | 1,660 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 5,099 | 452.9 |
| 512 | 7,160 | 149.1 |
| 2048 | 6,788 | 10.4 |
| 4096 | 6,605 | 1.8 |
| 8192 | 6,359 | 5.5 |
| 16384 | 5,846 | 5.2 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 122.3 | 0.2 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/gemma-4-12b-it-q6-k.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/gemma-4-12b-it-q6-k.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gemma-4-12b-it-q6-k.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gemma-4-12b-it-q6-k.md).*
