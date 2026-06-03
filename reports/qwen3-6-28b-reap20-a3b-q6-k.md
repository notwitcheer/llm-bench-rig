# Benchmark Report: Qwen3.6-28B-REAP20-A3B (Q6_K)

**Date:** 2026-06-03  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3.6-28B-REAP20-A3B |
| Parameters | 28.24 B (moe (3b active)) |
| Quantization | Q6_K |
| File size | 21.64 GiB |
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
| **MMLU** | **87.72%** | accuracy |
| **ARC-Challenge** | **95.00%** | accuracy |
| **HellaSwag** | **82.00%** | accuracy |
| **HumanEval** | **94.00%** | pass@1 |
| **GSM8K** | **90.00%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 97.22% | 35 / 36 |
| Humanities | 79.17% | 19 / 24 |
| Social Sciences | 96.15% | 25 / 26 |
| Other | 75.00% | 21 / 28 |

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 3,778 | 59.8 |
| 512 | 8,227 | 26.7 |
| 2048 | 8,163 | 46.2 |
| 4096 | 7,949 | 46.7 |
| 8192 | 7,685 | 29.9 |
| 16384 | 7,071 | 11.9 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 246.6 | 1.7 |

---

## Methodology

### Evaluation Framework

Custom generative evaluators built for this rig. All benchmarks run through llama-server's `/v1/chat/completions` endpoint.

- **Scoring:** Generative evaluation (not loglikelihood)
- **Thinking:** enabled
- **MCQ scoring:** First valid letter extracted from response (A/B/C/D)
- **Temperature:** 0 (deterministic)
- **Max tokens:** 2,048
- **GPU offload:** All layers (`-ngl 99`)

---

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-28b-reap20-a3b-q6-k.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-28b-reap20-a3b-q6-k.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-28b-reap20-a3b-q6-k.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-28b-reap20-a3b-q6-k.md).*
