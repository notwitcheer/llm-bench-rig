# Benchmark Report: Qwen3.6-35B-A3B (UD-Q4_K_M)

**Date:** 2026-06-04  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3.6-35B-A3B |
| Parameters | 34.66 B (moe (3b active)) |
| Quantization | UD-Q4_K_M |
| File size | 20.61 GiB |
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
| **MMLU** | **84.99%** | accuracy |
| **ARC-Challenge** | **95.73%** | accuracy |
| **HellaSwag** | **93.35%** | accuracy |
| **HumanEval** | **95.73%** | pass@1 |
| **GSM8K** | **96.66%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 84.27% | 1,270 / 1,507 |
| Humanities | 83.31% | 1,318 / 1,582 |
| Social Sciences | 91.59% | 1,514 / 1,653 |
| Other | 81.83% | 1,856 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 3,584 | 45.7 |
| 512 | 9,217 | 50.7 |
| 2048 | 9,004 | 47.0 |
| 4096 | 8,731 | 56.7 |
| 8192 | 8,347 | 21.4 |
| 16384 | 7,641 | 10.8 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 270.6 | 1.7 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-35b-a3b-ud-q4-k-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-6-35b-a3b-ud-q4-k-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-35b-a3b-ud-q4-k-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-6-35b-a3b-ud-q4-k-m.md).*
