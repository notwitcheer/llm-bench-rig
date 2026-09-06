# llm-bench-rig

Professional benchmarking pipeline for GGUF and safetensors models on NVIDIA GPUs. Built for RTX 5090 (Blackwell), works with any CUDA-capable GPU.

Dual-engine: **llama.cpp** for GGUF models, **vLLM** for safetensors. Produces structured JSON results, live web dashboard, and publishable benchmark cards.

## What it does

- **Speed benchmarks**: a llama-bench sweep, `-p 128,512,2048,4096,8192,16384 -n 128` (prompt processing at six context lengths plus tg128 generation, `speed.context_lengths` in config.yaml), or the vLLM API for safetensors models. A served (HTTP) lane, `scripts/speed_served.py`, times streaming chat completions against a running server and reports TTFT and perceived tokens per second as percentiles.
- **Quality benchmarks**: five board tasks (MMLU, ARC-Challenge, HellaSwag, GSM8K, HumanEval) run through our own generative evaluators in `lib/evals/` against llama-server chat completions, at temperature 0, letter or number extraction and code execution for scoring. There is no lm-evaluation-harness dependency, so multiple-choice scores are generative rather than loglikelihood based and can differ from harness numbers by several points; they are consistent with each other, which is what the board needs. GPQA-diamond is a second tier task: reported alongside, never part of `q_avg`.
- **Error bars**: `scripts/board_ci.py` writes `dataset/board_ci.csv` with a Wilson 95% interval per task per row and a propagated `q_avg` half-width, so neighbouring rows can be read as a tie when they should be.
- **Provenance**: every run writes a `provenance` block into `results/<slug>/meta.json` (llama-server build and chat template hash, server command line, gguf sha256, harness git sha, resolved quality config) so a number can be traced back to what produced it.
- **Live dashboard** — FastAPI + SSE, real-time GPU stats and benchmark progress
- **Export** — HTML reports, PNG cards (1200x675 for X/HF), cross-model comparison pages
- **Queue manager** — batch benchmarks with review gates between models

## Setup

```bash
# Clone and configure
cp config.example.yaml config.yaml
# Edit config.yaml with your paths and hardware

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

## Usage

```bash
# Benchmark a single model
python bench.py path/to/model.gguf

# Speed benchmarks only
python bench.py path/to/model.gguf --speed-only

# Queue multiple models
python bench_queue.py add path/to/model.gguf
python bench_queue.py add-all    # scan HuggingFace cache
python bench_queue.py list
python bench_queue.py start

# Live dashboard (run alongside benchmarks)
python dashboard.py

# Export results
python export.py model-slug
python export.py --compare model-a model-b model-c
```

## Requirements

- Python 3.10+
- CUDA-capable GPU with nvidia-smi
- llama.cpp built with CUDA (for GGUF models)
- vLLM (for safetensors models, optional)
- Playwright + Chromium (for PNG card export)

## Hardware tested

- NVIDIA RTX 5090 32GB (Blackwell, sm_120)
- CUDA 12.8 (patched for glibc 2.41 compatibility)
- Ubuntu 26.04 LTS

## License

MIT
