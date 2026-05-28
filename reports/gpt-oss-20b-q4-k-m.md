# Benchmark Report: gpt-oss-20b (Q4_K_M)

**Date:** 2026-05-28  
**Author:** WITCHEER  
**Platform:** RTX 5090 Benchmark Rig (capsule)  

---

## Model

| Field | Value |
|-------|-------|
| Model | gpt-oss-20b |
| Parameters | 20.91B (dense) |
| Quantization | Q4_K_M |
| File size | 10.83 GiB |
| Engine | llama.cpp (CUDA 12.8, sm_120) |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB GDDR7) |
| CPU | AMD Ryzen 5 9600 (6c/12t, 3.8/5.2 GHz) |
| RAM | 64 GB DDR5-5600 |
| OS | Ubuntu Server 26.04 LTS (headless) |
| CUDA | 12.8 (patched for glibc 2.41 compat) |

---

## Quality Benchmarks

All benchmarks use generative evaluation via llama-server chat completions. Multiple-choice tasks (MMLU, ARC, HellaSwag) use letter extraction instead of loglikelihood scoring — results are internally consistent for model comparison but absolute scores may differ from logprob-based evaluations by 5–15%.

### Summary

| Benchmark | Score | Metric | Correct / Total | Time |
|-----------|------:|--------|----------------:|-----:|
| **MMLU** (5-shot) | **78.56%** | accuracy | 11,031 / 14,042 | 3h 49m |
| **ARC-Challenge** (25-shot) | **94.62%** | accuracy | 1,109 / 1,172 | 10m 40s |
| **HellaSwag** (10-shot) | **74.49%** | accuracy | 7,480 / 10,042 | 3h 31m |
| **GSM8K** (5-shot, CoT) | **94.77%** | exact match | 1,250 / 1,319 | 22m 0s |
| **HumanEval** (0-shot) | **12.20%** | pass@1 | 20 / 164 | 2m 48s |

**Total evaluation time:** 7h 56m

### MMLU Breakdown by Category

| Category | Score | Correct / Total |
|----------|------:|----------------:|
| STEM | 89.83% | 2,711 / 3,018 |
| Social Sciences | 84.45% | 2,796 / 3,311 |
| Humanities | 77.45% | 2,456 / 3,171 |
| Other | 67.55% | 3,068 / 4,542 |

**Top 5 subjects:**

| Subject | Score |
|---------|------:|
| High School Computer Science | 99.0% |
| Elementary Mathematics | 96.8% |
| College Physics | 95.1% |
| High School Mathematics | 93.7% |
| College Biology | 92.4% |

**Bottom 5 subjects:**

| Subject | Score |
|---------|------:|
| Professional Law | 44.3% |
| Global Facts | 47.0% |
| Virology | 59.0% |
| Moral Disputes | 67.9% |
| Philosophy | 68.8% |

### Parse Reliability

The model uses extended reasoning (`reasoning_content` field) before responding. With `max_tokens=2048`, most reasoning chains complete successfully.

| Benchmark | Parse Failures | Failure Rate |
|-----------|---------------:|-------------:|
| MMLU | 653 | 4.6% |
| ARC-Challenge | 5 | 0.4% |
| HellaSwag | 37 | 0.4% |
| GSM8K | 0 | 0.0% |
| **Total** | **695** | **2.6%** |

Parse failures are scored as incorrect. The majority occur in MMLU subjects with long reasoning chains (professional_law, moral_scenarios) where the model's thinking exceeds the token budget.

---

## Speed Benchmarks

Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).

### Prompt Processing (tokens/s)

| Context Length | Speed | ±σ |
|---------------:|------:|---:|
| 128 | 7,221 | 67 |
| 512 | 16,750 | 149 |
| 2,048 | 13,524 | 12 |
| 4,096 | 11,685 | 44 |
| 8,192 | 9,414 | 16 |
| 16,384 | 6,678 | 14 |

### Generation (tokens/s)

| Metric | Speed | ±σ |
|--------|------:|---:|
| tg128 | 367.9 | 1.2 |

### Context Degradation

Prompt processing peaks at 512 tokens (16,750 t/s) then drops 60% at 16K context (6,678 t/s). This is the steepest degradation of any model in the rig — characteristic of smaller dense models with limited KV-cache efficiency.

---

## Methodology

### Evaluation Framework

Custom generative evaluators built for this rig. No dependency on `lm-evaluation-harness` — all benchmarks run through llama-server's `/v1/chat/completions` endpoint.

| Benchmark | Dataset | Eval Split | Few-shot | Scoring |
|-----------|---------|-----------|----------|---------|
| MMLU | `cais/mmlu` | test (14,042) | 5-shot per subject from `dev` | First valid A/B/C/D letter extracted from response |
| ARC-Challenge | `allenai/ai2_arc` | test (1,172) | 25-shot from `train` | First valid letter, numeric labels normalized to A–D |
| HellaSwag | `Rowan/hellaswag` | validation (10,042) | 10-shot from `train` | First valid A/B/C/D letter |
| GSM8K | `openai/gsm8k` | test (1,319) | 5-shot CoT from `train` | Exact match on extracted numeric answer |
| HumanEval | `openai/openai_humaneval` | test (164) | 0-shot | pass@1 via subprocess execution (10s timeout) |

### Inference Configuration

- **Server:** llama-server (llama.cpp, CUDA 12.8, Blackwell sm_120)
- **Temperature:** 0 (deterministic)
- **Max tokens:** 2,048 (accommodates reasoning models)
- **GPU offload:** All layers (`-ngl 99`)
- **Serving:** Single request, sequential (no batching)

### Differences from Standard Benchmarks

- **Generative vs loglikelihood:** MMLU, ARC, and HellaSwag are traditionally scored using token logprobabilities. This rig uses generative letter extraction, which typically yields scores 5–15% lower on the same model. Rankings between models remain consistent.
- **Thinking models:** gpt-oss-20b produces extended reasoning in a separate `reasoning_content` field. When the primary `content` field is empty, the evaluator falls back to parsing the reasoning chain for the final answer.
- **No normalized accuracy:** Standard HellaSwag reporting uses `acc_norm` (length-normalized). This rig reports raw accuracy, which may be lower for completions of varying length.

---

## Reproduction

```bash
# On capsule (192.168.1.9)
cd ~/benchmark-rig && source venv/bin/activate

# Full benchmark (speed + quality)
python3 bench.py /path/to/gpt-oss-20b-Q4_K_M.gguf

# Quality only
python3 bench.py /path/to/gpt-oss-20b-Q4_K_M.gguf --quality-only

# Individual evaluator
python3 -m lib.evals.mmlu --api-base http://127.0.0.1:8090/v1 --model gpt-oss-20b
```

All results, detailed per-subject breakdowns, and checkpoint files are stored in `results/gpt-oss-20b-q4-k-m/`.

---

*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. Source: [github.com/notwitcheer/llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig). Dataset: [huggingface.co/datasets/witcheer/rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks).*
