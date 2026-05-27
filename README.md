# llm-bench-rig

Professional benchmarking pipeline for GGUF and safetensors models on NVIDIA GPUs. Built for RTX 5090 (Blackwell), works with any CUDA-capable GPU.

Dual-engine: **llama.cpp** for GGUF models, **vLLM** for safetensors. Produces structured JSON results, live web dashboard, and publishable benchmark cards.

## What it does

- **Speed benchmarks** — prompt processing (pp128/pp512/pp2048) and text generation (tg128) via llama-bench or vLLM API
- **Quality benchmarks** — MMLU, ARC-Challenge, HellaSwag, HumanEval, GSM8K via lm-evaluation-harness
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
