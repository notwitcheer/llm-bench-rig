# Two 120-billion-parameter reasoning models on one consumer card

**Rig:** RTX 5090 32GB (capsule) · **Date:** 2026-06-08 · **Task:** local-ai-roadmap t035
**Model:** Mistral Small 4 (119B / 6B-active MoE, 128 experts/4 active, MXFP4_MOE GGUF) vs gpt-oss-120B (117B / 5.1B-active, MXFP4)

## The point

Mistral Small 4 is a 119-billion-parameter model. The card has 32GB of VRAM. It runs anyway — and not as a
party trick: **~36 tokens/sec of usable generation.** That's the whole wedge of this rig — data-center-class
models on one buyable card — so the question is how, and how it stacks against the only comparable thing we've
run: gpt-oss-120B, the other ~120B reasoning MoE that fits this lane.

## Speed, head to head

Both quantized to MXFP4, both with MoE experts streamed from system RAM (`--n-cpu-moe`), attention + active
path on the GPU. `llama-bench`, RTX 5090 32GB:

| | Mistral Small 4 | gpt-oss-120B |
|---|---|---|
| total / active params | 119B / 6B | 117B / 5.1B |
| GGUF size | 66.9 GB | 59.0 GB |
| offload | `n_cpu_moe 24` | `n_cpu_moe 20` |
| VRAM used | 27.5 GB | ~30 GB |
| **decode (tg)** | **35.95 tok/s** | **46.48 tok/s** |
| **prefill peak (pp)** | **313 tok/s** | **588 tok/s** |

gpt-oss is the faster of the two by ~30% on decode and nearly 2x on prefill. It isn't a better engine — it's a
**leaner** one: 8GB smaller, ~1B fewer active params, and two fewer layers' worth of experts offloaded. Every
one of those differences pushes the same direction.

## The mechanism — why a 119B model decodes at a usable rate at all

Decode is memory-bound, and on an offloaded MoE the bound is the *active* path, not the total weight. Only **6B
of the 119B fire per token**, so each token reads a small slice of expert weights from RAM, not the whole model.
That's why ~36 tok/s is achievable despite 40GB of the model living in system RAM. The cost is borne on **prefill**
(313 vs gpt-oss's 588 tok/s) and on absolute decode (36 vs 46) — both tracking Mistral's larger size and heavier
offload. Bigger model, more offload, slower — but the active-params trick keeps it the right side of usable.

## Feasibility notes (for anyone trying this on Blackwell)

- **Arch `mistral4` loads on llama.cpp b9365** — no rebuild needed (it was an open question; March-2026 arch).
- **There is no NVFP4 GGUF.** The official NVFP4 checkpoint is vLLM/compressed-tensors only, which is walled on
  this toolkit-less box (the t036 wall). The runnable path is the **MXFP4_MOE GGUF** (unsloth) — which, bonus,
  makes the gpt-oss head-to-head a clean same-quant comparison.
- **Fit:** `n_cpu_moe 24`, ~27.5GB VRAM, ~46GB RAM, this build's auto-fit ("fitting params to device memory")
  handles the split. Donald (the resident agent) must vacate the GPU for the run.

## Why no quality leaderboard entry (the honest part)

This was meant to be a Thinking-ON leaderboard entry. It isn't — and the reason is a real lesson. **The rig's
think toggle is Qwen-shaped:** it engages reasoning by *omitting* `/nothink` and relies on the model thinking by
default. Mistral Small 4 doesn't — it gates reasoning behind a `reasoning_effort` dial that defaults to `none`.
So the model ran in near-non-reasoning mode (~5 sec/question, ~110-190 tokens — far too short for real
chain-of-thought), and its scores would be a no-think measurement mislabeled as think-ON. Invalid; dropped.

Fixing it is a small harness change (send `reasoning_effort: high` when think is on). But a *properly* reasoning
run is slow on this offloaded box — real chain-of-thought at 36 tok/s makes even a 6%-sampled MMLU a multi-hour
job. So a valid Thinking-ON quality entry is deferred to a dedicated run; this treatment ships the speed and
feasibility finding, which is the part that's solid.

---
Speed data: `results/mistral-small-4-119b-2603-mxfp4-moe/speed.json`. Harness: the rig's standard `bench.py`
speed sweep with the per-model offload override (`offload:` in config). gpt-oss-120B figures from its prior treatment.
