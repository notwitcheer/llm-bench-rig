# Benchmark Report: Qwen3.6-35B-A3B (UD-Q6_K)

**Date:** 2026-06-03  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3.6-35B-A3B |
| Parameters | 34.66 B (moe (3b active)) |
| Quantization | UD-Q6_K |
| File size | 27.3 GiB |
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
| **MMLU** | **94.74%** | accuracy |
| **ARC-Challenge** | **97.00%** | accuracy |
| **HellaSwag** | **87.00%** | accuracy |
| **HumanEval** | **98.00%** | pass@1 |
| **GSM8K** | **92.00%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 100.00% | 36 / 36 |
| Humanities | 87.50% | 21 / 24 |
| Social Sciences | 96.15% | 25 / 26 |
| Other | 92.86% | 26 / 28 |

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 3,048 | 35.2 |
| 512 | 8,321 | 54.8 |
| 2048 | 8,125 | 51.1 |
| 4096 | 7,901 | 51.1 |
| 8192 | 7,599 | 26.1 |
| 16384 | 6,998 | 20.6 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 259.6 | 1.5 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-35b-a3b-ud-q6-k.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-35b-a3b-ud-q6-k.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-35b-a3b-ud-q6-k.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-35b-a3b-ud-q6-k.md).*
