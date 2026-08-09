# Benchmark Report: Ling-3.0-flash (Q3_K_M)

**Date:** 2026-08-09  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Ling-3.0-flash |
| Parameters | 127.49 B (dense) |
| Quantization | Q3_K_M |
| File size | 58.27 GiB |
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
| **MMLU** | **83.95%** | accuracy |
| **ARC-Challenge** | **96.33%** | accuracy |
| **HellaSwag** | **93.17%** | accuracy |
| **HumanEval** | **92.07%** | pass@1 |
| **GSM8K** | **93.48%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 85.53% | 1,289 / 1,507 |
| Humanities | 82.30% | 1,302 / 1,582 |
| Social Sciences | 90.50% | 1,496 / 1,653 |
| Other | 79.28% | 1,798 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 73.0 | 15.9 |
| 512 | 261.1 | 14.5 |
| 2048 | 290.1 | 2.4 |
| 4096 | 298.4 | 1.2 |
| 8192 | 300.4 | 1.0 |
| 16384 | 302.5 | 0.9 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 46.0 | 0.2 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-q3-k-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/ling-3-0-flash-q3-k-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-q3-k-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ling-3-0-flash-q3-k-m.md).*
