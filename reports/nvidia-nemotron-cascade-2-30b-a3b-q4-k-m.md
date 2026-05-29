# Benchmark Report: nvidia_Nemotron-Cascade-2-30B-A3B (Q4_K_M)

**Date:** 2026-05-29  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | nvidia_Nemotron-Cascade-2-30B-A3B |
| Parameters | 31.58 B (moe (3b active)) |
| Quantization | Q4_K_M |
| File size | 23.03 GiB |
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
| **MMLU** | **74.42%** | accuracy |
| **ARC-Challenge** | **91.55%** | accuracy |
| **HellaSwag** | **75.68%** | accuracy |
| **HumanEval** | **79.27%** | pass@1 |
| **GSM8K** | **87.11%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 69.34% | 1,045 / 1,507 |
| Humanities | 74.84% | 1,184 / 1,582 |
| Social Sciences | 83.79% | 1,385 / 1,653 |
| Other | 70.68% | 1,603 / 2,268 |

*Sampled at 50% (seed 42)*

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 4,397 | 23.7 |
| 512 | 10,647 | 83.4 |
| 2048 | 10,338 | 32.8 |
| 4096 | 10,029 | 25.0 |
| 8192 | 9,508 | 20.6 |
| 16384 | 8,580 | 9.8 |
| 32768 | 6,968 | 12.5 |
| 65536 | 4,967 | 7.0 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 350.8 | 1.4 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/nvidia-nemotron-cascade-2-30b-a3b-q4-k-m.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/nvidia-nemotron-cascade-2-30b-a3b-q4-k-m.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/nvidia-nemotron-cascade-2-30b-a3b-q4-k-m.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/nvidia-nemotron-cascade-2-30b-a3b-q4-k-m.md).*
