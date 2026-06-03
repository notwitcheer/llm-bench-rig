# Benchmark Report: gpt-oss-120b (mxfp4)

**Date:** 2026-06-03  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | gpt-oss-120b |
| Parameters | 116.83 B (dense) |
| Quantization | mxfp4 |
| File size | 0.01 GiB |
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
| **MMLU** | **89.47%** | accuracy |
| **ARC-Challenge** | **95.00%** | accuracy |
| **HellaSwag** | **80.00%** | accuracy |
| **HumanEval** | **98.00%** | pass@1 |
| **GSM8K** | **97.00%** | exact_match |

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| Stem | 97.22% | 35 / 36 |
| Humanities | 83.33% | 20 / 24 |
| Social Sciences | 88.46% | 23 / 26 |
| Other | 85.71% | 24 / 28 |

---

## Speed Benchmarks

Measured with `llama-bench`. MoE experts offloaded to CPU (`--n-cpu-moe 20`), attention/active path on GPU (`-ngl 99`). Model is 59GB and does not fit 32GB VRAM; tuned to ~30GB VRAM with the rest in system RAM.

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 124.5 | 33.1 |
| 512 | 484.4 | 37.3 |
| 2048 | 588.2 | 15.0 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 46.5 | 0.4 |

---

## Methodology

### Evaluation Framework

Custom generative evaluators built for this rig. All benchmarks run through llama-server's `/v1/chat/completions` endpoint.

- **Scoring:** Generative evaluation (not loglikelihood)
- **Thinking:** enabled
- **MCQ scoring:** First valid letter extracted from response (A/B/C/D)
- **Temperature:** 0 (deterministic)
- **Max tokens:** 2,048 (MCQ/GSM8K); HumanEval 4,096, no stop sequences (reasoning models emit code after long inline reasoning)
- **GPU offload:** MoE experts of first 20 layers on CPU (`--n-cpu-moe 20`); attention + active path on GPU. ~30GB of 32GB VRAM, remainder in RAM.

---

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig/blob/main/reports/gpt-oss-120b-mxfp4.md](https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/gpt-oss-120b-mxfp4.md). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gpt-oss-120b-mxfp4.md](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/gpt-oss-120b-mxfp4.md).*
