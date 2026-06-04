# Benchmark Report: gpt-oss-20b (Q4_K_M)

**Date:** 2026-06-04  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | gpt-oss-20b |
| Parameters | 20.91 B (dense) |
| Quantization | Q4_K_M |
| File size | 10.83 GiB |
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
| **MMLU** | **78.56%** | accuracy |
| **ARC-Challenge** | **94.62%** | accuracy |
| **HellaSwag** | **74.49%** | accuracy |
| **HumanEval** | **94.51%** | pass@1 |
| **GSM8K** | **94.77%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 89.83% | 2,711 / 3,018 |
| Humanities | 77.45% | 2,456 / 3,171 |
| Social Sciences | 84.45% | 2,796 / 3,311 |
| Other | 67.55% | 3,068 / 4,542 |

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 7,172 | 73.0 |
| 512 | 16,646 | 117.8 |
| 2048 | 13,456 | 36.4 |
| 4096 | 11,684 | 28.1 |
| 8192 | 9,408 | 22.7 |
| 16384 | 6,669 | 4.6 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 367.4 | 0.9 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/gpt-oss-20b-q4-k-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/gpt-oss-20b-q4-k-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gpt-oss-20b-q4-k-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gpt-oss-20b-q4-k-m.md).*
