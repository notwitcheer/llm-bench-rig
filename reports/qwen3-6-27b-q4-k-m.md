# Benchmark Report: Qwen3.6-27B (Q4_K_M)

**Date:** 2026-05-29  
**Author:** WITCHEER  
**Platform:** NVIDIA GeForce RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | Qwen3.6-27B |
| Parameters | 26.90 B (dense) |
| Quantization | Q4_K_M |
| File size | 15.66 GiB |
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

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | +/-sigma |
|---------------:|------:|---------:|
| 128 | 2,973 | 322.8 |
| 512 | 3,826 | 41.6 |
| 2048 | 3,741 | 1.3 |
| 4096 | 3,645 | 2.8 |
| 8192 | 3,485 | 7.2 |
| 16384 | 3,162 | 3.7 |

### Generation (tokens/s)

| Metric | Speed | +/-sigma |
|--------|------:|---------:|
| tg128 | 77.1 | 0.2 |

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

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks).*
