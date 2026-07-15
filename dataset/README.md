---
license: mit
task_categories:
  - text-generation
tags:
  - benchmark
  - inference
  - llm
  - nvidia
  - rtx-5090
  - llama-cpp
  - vllm
  - speed
  - quality
  - mmlu
  - gsm8k
  - humaneval
  - moe
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: benchmarks.csv
---

# RTX 5090 LLM Benchmarks

Speed and quality benchmarks for quantized LLMs on NVIDIA RTX 5090 32GB, measured with [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig).

## Quality Benchmarks

Generative evaluation through llama-server chat completions. Replicates standard benchmark methodology using custom evaluators — no `lm-evaluation-harness` dependency.

Results are **split by reasoning mode**: comparing a thinking-on (reasoning) model's quality against a thinking-off model is apples-to-oranges, so the two groups are ranked separately. `q_avg` is the mean of the five tasks.

### Thinking OFF (non-reasoning · direct answer)

| Model | Params | Quant | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | q_avg |
|-------|-------:|-------|-----:|------:|----------:|------:|----------:|------:|
| Gemma 4 31B-it | 30.70B | Q6_K | 87.8 | 97.6 | 92.0 | 97.5 | 96.3 | **94.2** |
| Qwen3.6-27B² | 26.90B | Q6_K | 87.9 | 96.9 | 95.4 | 97.3 | 93.3 | **94.2** |
| Qwen3.6-27B² | 26.90B | NVFP4³ | 87.6 | 96.4 | 95.4 | 97.5 | 93.3 | **94.1** |
| Qwen3.6-35B-A3B | 34.66B | UD-Q4_K_M | 85.0 | 95.7 | 93.3 | 96.7 | 95.7 | **93.3** |
| Qwen3-Coder-Next | 79.67B | UD-Q2_K_XL | 83.7 | 96.0 | 89.3 | 96.0 | 93.3 | **91.7** |
| Gemma 4 12B-it | 11.91B | Q6_K | 78.9 | 94.0 | 81.6 | 96.4 | 87.2 | **87.6** |
| gpt-oss-20b | 20.91B | Q4_K_M | 78.6 | 94.6 | 74.5 | 94.8 | 94.5 | **87.4** |
| Nemotron-Cascade-2 | 31.58B | Q4_K_M | 74.4 | 91.5 | 75.7 | 87.1 | 79.3 | **81.6** |

> ² Re-banked 2026-07-15 under the pinned harness (rig ec00ff0, llama-server b9653); the re-bank moved HumanEval by one passing problem (92.7 → 93.3) and nothing else beyond 0.05. ³ Served via vLLM 0.21.0 (native cutlass sm_120 FP4 path, lm_head dequanted); GGUF rows serve via llama-server. Full protocol and the per-suite delta table: [`reports/qwen3-6-27b-nvfp4.md`](../reports/qwen3-6-27b-nvfp4.md).

### Thinking ON (reasoning · extended chain-of-thought)

| Model | Params | Quant | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | q_avg |
|-------|-------:|-------|-----:|------:|----------:|------:|----------:|------:|
| Qwen3.6-35B-A3B | 34.66B | UD-Q6_K | 94.7 | 97.0 | 87.0 | 92.0 | 98.0 | **93.8** |
| gpt-oss-120B¹ | 116.83B | MXFP4 | 89.5 | 95.0 | 80.0 | 97.0 | 98.0 | **91.9** |
| Qwen3.6-28B-REAP-A3B | 28.24B | Q6_K | 87.7 | 95.0 | 82.0 | 90.0 | 94.0 | **89.7** |

> **HumanEval correction (2026-06-04).** An earlier harness passed API stop sequences (`\ndef`, `\nclass`) that fired *mid-reasoning*, truncating inline-reasoning models before they emitted code — producing false-low scores (Qwen3-Coder-Next read **10%**, not 93%). Every model has since been re-run on the fixed, reasoning-aware harness (no stop sequences, `max_tokens=4096`, indentation-preserving response handling). **Do not cite any HumanEval figure published before this date.**
>
> **Why two tables.** Thinking-off rows answer directly; thinking-on rows emit an extended reasoning chain first. The two modes are not comparable on the same axis — including on MCQ/GSM8K — so they are ranked separately. Within a family, turning thinking on trades raw knowledge recall for reasoning depth (compare Qwen3.6-35B-A3B in both tables: MMLU 85.0 → 94.7).
>
> ¹ **gpt-oss-120B** runs via MoE CPU-offload (`--n-cpu-moe 20`) — it does not fit 32GB VRAM (59GB model); ~30GB VRAM + the rest in system RAM, ~47 tok/s generation. It and the other two thinking-on rows were run on a ~100-item-per-task subset (MMLU 2/subject).

> **Sampling.** MMLU & HellaSwag use 50% stratified sampling (seed=42); ARC-Challenge, GSM8K, and HumanEval run the full item counts (HumanEval = all 164). Full per-model reports in [`reports/`](reports/).

### Methodology

| Benchmark | Dataset | Few-shot | Scoring | Items |
|-----------|---------|----------|---------|------:|
| MMLU | `cais/mmlu` | 5-shot | Letter extraction (A/B/C/D) | 14,042 |
| ARC-Challenge | `allenai/ai2_arc` | 25-shot | Letter extraction | 1,172 |
| HellaSwag | `Rowan/hellaswag` | 10-shot | Letter extraction | 10,042 |
| GSM8K | `openai/gsm8k` | 5-shot CoT | Exact numeric match | 1,319 |
| HumanEval | `openai/openai_humaneval` | 0-shot | pass@1 (code execution) | 164 |

All benchmarks run at `temperature=0`. MCQ and GSM8K use `max_tokens=2048`; HumanEval uses `max_tokens=4096` with **no stop sequences** (reasoning models emit code only after long inline reasoning — premature stops were the bug corrected above). Multiple-choice tasks use generative letter extraction instead of loglikelihood scoring — scores are internally consistent for model comparison but may differ from logprob-based evaluations by 5-15%.

Full per-model reports with MMLU category breakdowns, parse reliability stats, and speed data: [`reports/`](reports/)

---

## Speed Benchmarks

### What's measured

- **Prompt processing (pp)**: parallel batched token throughput at context lengths 128, 512, 2048, 4096, 8192, 16384
- **Text generation (tg)**: sequential autoregressive token throughput at 128 tokens
- All models fully GPU-offloaded (ngl=99)

### Speed data schema

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `architecture` | Dense or MoE (with active param count) |
| `params_b` | Total parameters in billions |
| `quant` | Quantization method |
| `size_gib` | File size in GiB |
| `engine` | Inference engine (llama.cpp or vLLM) |
| `backend` | Compute backend (CUDA) |
| `gpu` | GPU model |
| `vram_gb` | VRAM in GB |
| `test` | Benchmark test (pp128, pp512, ..., tg128) |
| `tokens_per_sec` | Throughput in tokens/second |
| `stddev` | Standard deviation |
| `date` | Benchmark date |

### Key findings

MoE (3B active) vs Dense (27B) on same-family Qwen3.6 models:
- Prompt processing: **2.4x faster** across all context lengths
- Text generation: **3.5x faster** (271 vs 77 t/s)
- Both degrade ~17% at 16K context (attention + VRAM, not parameter count)

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 32GB (Blackwell, sm_120a) |
| CPU | AMD Ryzen 5 9600 (6c/12t) |
| RAM | 64GB DDR5-5600 |
| OS | Ubuntu 26.04 LTS |
| CUDA | 12.8 (patched for glibc 2.41) |

## Tooling

All benchmarks generated with [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig) — open-source pipeline for speed and quality benchmarks on GGUF and safetensors models.
