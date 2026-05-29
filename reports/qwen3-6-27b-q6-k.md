# Benchmark Report: Qwen3.6-27B (Q6_K)

**Date:** 2026-05-29  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3.6-27B |
| Parameters | 26.90 B (dense) |
| Quantization | Q6_K |
| File size | 20.98 GiB |
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
| **MMLU** | **87.92%** | accuracy |
| **ARC-Challenge** | **96.93%** | accuracy |
| **HellaSwag** | **95.44%** | accuracy |
| **HumanEval** | **18.90%** | pass@1 |
| **GSM8K** | **97.27%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 87.26% | 1,315 / 1,507 |
| Humanities | 89.51% | 1,416 / 1,582 |
| Social Sciences | 93.22% | 1,541 / 1,653 |
| Other | 83.38% | 1,891 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 2,560 | 230.8 |
| 512 | 3,191 | 32.2 |
| 2048 | 3,153 | 10.0 |
| 4096 | 3,079 | 3.6 |
| 8192 | 2,956 | 2.0 |
| 16384 | 2,725 | 0.8 |
| 32768 | 2,333 | 9.1 |
| 65536 | 1,772 | 0.6 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 61.8 | 0.1 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-27b-q6-k.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-27b-q6-k.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-27b-q6-k.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-27b-q6-k.md).*
