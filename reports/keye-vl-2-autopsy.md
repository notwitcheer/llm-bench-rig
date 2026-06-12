# Keye-VL-2.0-30B on a consumer GPU: an autopsy — five measured walls between "open weights" and a running model

**Rig:** one RTX 5090 32GB · 64GB RAM · CUDA 12.8 (sm_120) · transformers 4.57.1 through 5.11 + 5.0.0rc0-rc3 tested
**Model:** [Kwai-Keye/Keye-VL-2.0-30B-A3B](https://huggingface.co/Kwai-Keye/Keye-VL-2.0-30B-A3B) (Apache 2.0, 30B/3B-active MoE, 62.3GB bf16) — Kuaishou's long-video VLM ([arXiv 2606.10651](https://arxiv.org/abs/2606.10651)): DeepSeek-style sparse attention adapted to a GQA multimodal stack, claiming **"lossless" 256K context / hour-level video**. Released ~2 weeks ago; zero community local-run reports existed before this attempt. This is what happened when one tried.

## Wall 1: the 256K claim is dead on arrival at 32GB — by arithmetic

DSA here is **compute** sparsity, not KV compression: the full GQA KV cache is stored.
48 layers x 2 x 4 KV heads x 128 dim x bf16 = **96 KiB/token → 25.8GB at 256K**, before
a single weight. With 62.3GB of bf16 weights and no quant path (wall 3), no configuration
of this model reaches 256K on any consumer card. The headline claim is datacenter-only.

## Wall 2: the "sparse" attention is O(N²)-memory on consumer stacks — measured

The official fast path (forked SGLang + DeepGEMM + custom kernels, H800 x2 reference
config) does not build on sm_120 — DeepGEMM has no consumer-Blackwell support. The shipped
fallback is a pure-PyTorch reference implementation of DSA, and its indexer **materializes
the full N x N score matrix** (16 heads, bf16 = 32·N² bytes) before top-k:

| context | indexer scores | KV cache |
|---|---|---|
| 8K | 2.1GB | 0.8GB |
| 32K | **32.9GB — measured: one 30.65GiB allocation, OOM** | 3.2GB |
| 256K | 2.1TB | 25.8GB |

A 60-second video prompt (~32K tokens) OOMs a fully-drained 32GB card on the *indexer
scoring step alone*. The sparse-attention model is **more** memory-hungry than dense
attention everywhere its custom kernels don't exist.

## Wall 3: 4-bit quantization reaches 4.7% of the model — measured

The experts are stored fused — 3D `nn.Parameter` tensors (`[128, 2048, 1536]` per layer),
not `nn.Linear` — so every consumer quantizer (bnb, torchao) walks right past them:
**452 Linear modules = 1.46B of 31.12B params (4.7%) is all that 4-bit can touch.**
No official quants exist, llama.cpp has no arch support (and no GGUFs), vLLM/SGLang
mainline don't register `KeyeVL2`, so the AWQ route is closed too. The only community
conversions that reportedly run are MLX 4-bit on Apple silicon (untested here).
On CUDA, bf16 + CPU offload is the only load that exists: 58GB footprint, 0.5 tok/s.

## Wall 4: the code targets a one-week transformers API window

The trust_remote_code modeling file requires, simultaneously: `OutputRecorder` (removed
in 5.2), factory-form `check_model_inputs` (absent in 4.57), `SlidingWindowCache`
(removed in 5.0.0rc3), and the pre-flip fused-expert layout (changed in 5.0.0rc2).
The intersection: **transformers 5.0.0rc0 / rc1 — two release candidates** — and nothing
else, before or since. On stable 5.0/5.11 it took six compatibility shims (a pure-PyTorch
Hadamard stand-in for the unbuildable `fast-hadamard-transform`, a metadata-complete
`flash_attn` stub to pass a module-level availability assert, dead-import cache classes,
a rope-init function, config attribute patches, a rotary-class method) plus a 62GB
expert-tensor re-layout just to reach a forward pass.

## Wall 5: it still doesn't work — and the diagnosis trail is exhaustive

On rc1, with the untouched original checkpoint and native semantics, generation is
incoherent noise. The per-layer trace shows **healthy hidden-state norms through all 48
layers** with garbage logits — well-scaled but semantically destroyed computation.
Systematically exonerated: the DSA path (the built-in dense fallback is equally broken),
expert orientation (dimensionally provable — gate_up must map 2048→1536, no transpose
ambiguity exists), gate/up chunk order (both tested), rope theta (10M, confirmed loaded),
M-RoPE text positions (explicit equals default), causal-mask alignment. Untried: pure-CPU
inference, transformers git-commit archaeology, and the official Hopper-only docker —
each past the point of reasonable cost.

**Verdict: thirteen run attempts deep, Keye-VL-2.0-30B does not produce coherent output
on consumer hardware through any reachable configuration.** For contrast, on this same
rig: HRM-Text-1B ran after one missing tensor was supplied; LocateAnything-3B ran after
one config knob. The gap between "open weights" and "runnable weights" has never measured
wider.

## Honest caveats

- A subtle interaction with our compat shims cannot be 100% excluded as the noise source —
  though the text path exercises none of their math (their attention is hand-rolled eager,
  the Hadamard rotation only feeds top-k *selection*, and the dense fallback bypasses it
  entirely and is equally broken).
- "Lossless 256K" itself was never benchmarkable here; the paper publishes no
  needle-in-video curves either — the claim rests on benchmark aggregates.
- The model may be excellent inside its intended habitat (H800 pairs, their docker, their
  SGLang fork). Every wall above is about the release engineering, not the research.

## Reproduce

`scripts/keye_probe.py` (shims + smoke + quant verdict) · `scripts/keye_convert.py`
(expert re-layout, unnecessary on rc0/rc1) · `scripts/keye_diag.py` (layer-norm trace) ·
`scripts/chart_keye_walls.py`. Environment: uv venv, torch 2.11+cu128,
`transformers==5.0.0rc1`, bf16, `device_map=auto` (28GiB GPU / 42GiB CPU).
