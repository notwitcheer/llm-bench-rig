# Benchmark Report: Qwen3-Coder-Next (UD-Q2_K_XL)

**Date:** 2026-05-29  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3-Coder-Next |
| Parameters | 79.67 B (moe) |
| Quantization | UD-Q2_K_XL |
| File size | 24.92 GiB |
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
| **MMLU** | **83.69%** | accuracy |
| **ARC-Challenge** | **95.99%** | accuracy |
| **HellaSwag** | **89.32%** | accuracy |
| **HumanEval** | **10.37%** | pass@1 |
| **GSM8K** | **95.98%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 84.07% | 1,267 / 1,507 |
| Humanities | 83.38% | 1,319 / 1,582 |
| Social Sciences | 90.62% | 1,498 / 1,653 |
| Other | 78.62% | 1,783 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 2,378 | 29.9 |
| 512 | 4,433 | 39.1 |
| 2048 | 4,417 | 17.8 |
| 4096 | 4,372 | 6.7 |
| 8192 | 4,254 | 18.4 |
| 16384 | 4,022 | 7.2 |
| 32768 | 3,562 | 2.1 |
| 65536 | 2,909 | 2.5 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 224.6 | 1.7 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-coder-next-ud-q2-k-xl.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/qwen3-coder-next-ud-q2-k-xl.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-coder-next-ud-q2-k-xl.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/qwen3-coder-next-ud-q2-k-xl.md).*
