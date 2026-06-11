# Block diffusion vs its autoregressive twin on one RTX 5090: the hyped speedup doesn't show — AR wins at every length

**Rig:** one RTX 5090 32GB · llama.cpp diffusion-gemma build (PR #24427, sm_120 CUDA) ·
**gemma-4-26B-A4B-it** (autoregressive) vs **diffusiongemma-26B-A4B-it** (block diffusion) ·
both **Q4_K_M**, both **source-converted through the same pipeline** · batch 1 · temp 0.4

**Setup:** These are the same Gemma-4 26B-A4B checkpoint family — one trained autoregressive, one as a
block-diffusion variant — so the only variable is the *generation paradigm*. Both were converted from the
original `google/…` safetensors with the **same** `convert_hf_to_gguf.py` and **plain** `Q4_K_M` (no imatrix),
run on the **same** llama.cpp build, same GPU. AR is served via `llama-server` (reasoning off → clean final
answers); diffusion via `llama-diffusion-gemma-cli` (128 steps, entropy-bound early-stop to ~48). Throughput
is each runtime's own **load-excluded** generation timing. **Effective answer tok/s = answer_tokens /
generation_seconds** — the fair cross-paradigm metric, because diffusion pays a *fixed* canvas cost
regardless of how long the answer is.

## The board

| prompt | answer len | AR tok/s | diffusion **eff** tok/s | diffusion raw canvas tok/s | AR faster by |
|---|---|---|---|---|---|
| factual-short | 2 | 114\* | 0.8 | 104 | **142×** |
| factual | 32 | 242 | 9.0 | 72 | **27×** |
| math | 16 | 249 | 4.5 | 73 | **55×** |
| reasoning | 256 | 248 | 107.6 | 107 | **2.3×** |
| explain-med | 256 | 245 | 72.5 | 72 | **3.4×** |
| generate-long | 256 | 251 | 71.9 | 72 | **3.5×** |

\* factual-short's AR rate is cold-start / 3-token measurement noise; AR steady-state is ~248 tok/s.
**VRAM: AR 17.6 GB · diffusion 19.6 GB.**

## The finding

**On its own autoregressive twin — identical training family, params, quant, hardware — block diffusion is
slower at every answer length, and uses more VRAM.** AR holds a rock-steady **~248 tok/s**. Diffusion pays a
fixed **~2.5–3.5 s** cost to denoise a 256-token canvas no matter how short the answer is, so:

- **Short answers are catastrophic** — a 2-token reply ("Canberra") still costs the whole canvas: **0.8 eff
  tok/s vs ~240** (≈140–300× slower).
- **Even its best case loses** — a reasoning answer that *fills* the 256 canvas peaks at **107 tok/s, still
  2.3× slower than AR**.
- **The raw canvas rate (72–108 tok/s) never reaches AR's ~248 either** — so the gap isn't only wasted canvas;
  the paradigm is intrinsically slower at this scale.

The widely-shared day-0 claim ("~120+ t/s, *faster* than autoregressive") **does not reproduce** on a single
5090 at Q4_K_M, batch 1.

## Quality

Both answer the factual prompts correctly (Canberra; Leonardo da Vinci, ~1503). In no-think mode **AR produced
clean, correct, complete answers on all six — including the bat-and-ball trap (`$0.05`, correct, with no
thinking channel)**. Diffusion **can't** be put in no-think mode (the CLI exposes no thinking control), so on
the four reasoning-heavy prompts it dumped raw `<|channel>thought` into the canvas and never resolved a clean
final answer (math truncated mid-formula at 16 tokens). So in practice AR was both faster *and* cleaner — with
the honest caveat that the diffusion CLI's lack of no-think control handicaps it on those prompts.

## Why diffusion might still matter (just not here)

Block diffusion's theoretical win is **parallelism** — denoising many tokens at once should amortize on long
outputs and, especially, under heavy **batching**. This test is **batch 1** on one GPU — the worst case for
it. The honest read: any throughput advantage needs serving load this rig didn't test, or more mature kernels
(PR #24427 is a draft). What's *measured*: at the single-stream interactive scale most people run locally,
autoregressive wins decisively.

## Caveats

- Single RTX 5090, **batch 1**, Q4_K_M, temp 0.4 — no batching/concurrency (diffusion's best case is untested).
- diffusion-gemma support is an **unmerged draft PR (#24427)** with an experimental sampler — kernels may improve.
- Both models were **converted from source** with the PR's own converter, because *every* public GGUF (unsloth,
  AlexAtomic, …) is missing the PR's self-conditioning tensors and won't load.
- AR forced no-think (clean answers); diffusion has **no** no-think control — the quality comparison is
  asymmetric on reasoning prompts.

## Worth it?

For **local single-stream use on a 5090 today: autoregressive Gemma-4 is the pick** — 2–300× faster and
cleaner output. Block diffusion is a genuinely interesting paradigm worth re-testing under batching and once
the kernels mature — but the day-0 "faster than AR" hype isn't real at this scale.

**Sources**
- llama.cpp diffusion-gemma PR: https://github.com/ggml-org/llama.cpp/pull/24427
- google/diffusiongemma-26B-A4B-it: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- runner + chart: https://github.com/notwitcheer/llm-bench-rig (`scripts/diff_vs_ar.py`, `scripts/chart_diff_vs_ar.py`)
