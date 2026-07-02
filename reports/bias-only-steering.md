# Bias-Only Reasoning Steering on one RTX 5090: nothing moves at bounded budget — and random rewards match correct ones

**t074 · 2026-07-02 · Qwen2.5-Math-7B · RLOO · bounded-budget reproduction + the controls the paper omits**

Paper: *Steering LLM Reasoning Through Bias-Only Adaptation* (arXiv 2505.18706, EMNLP 2025).
Companion: *Small Vectors, Big Effects* (arXiv 2509.06608). Code: corl-team/steering-reasoning.

## The claims

1. **Capability:** train ONE d-dim bias vector per layer (added to `mlp.down_proj` output,
   ≈0.0016% of params) with RL and match full RL fine-tuning. Their Table 1, Qwen2.5-Math-7B:
   MATH500 base 52.2 → full-FT 79.3 → **steering 79.9**; AMC23 45.8 → 64.2 → 62.5.
2. **Efficiency:** "34s vs 52m" training, "0.11s vs 9.94s" per step, 240KB vs 13.8GB optimizer
   memory (Qwen2.5-14B figures).
3. **Their own mechanism paper:** the last-layer vector "behaves like first-token
   substitution" — prefixing the token "To" recovers ~10–11 points on this base, "about
   three quarters" of the last-layer vector's gain.

## What we ran (and why it's not their code)

**Their pinned stack cannot execute on consumer Blackwell.** torch 2.6.0+cu124 (with
vllm 0.8.5.post1 + flashinfer cu124 on top) fails the first kernel launch on sm_120:
`CUDA error: no kernel image is available for execution on the device`. The stack is dead
on arrival on an RTX 5090, as shipped.

So we reimplemented their recipe on the rig's stack (torch 2.10+cu128, HF transformers),
pinned from their own configs (`configs/train/rl/qwen2.5-math-7b/deepscaler/`): RLOO,
temperature 1.0, steering lr 1e-3, their LoRA baseline config verbatim (rank 4, alpha 4,
`down_proj` only, lr 1e-4), `qwen_math` template, DeepScaleR dataset.

**Bounded budget, identical across arms** (disclosed deviation from their full recipe of
~1 epoch ≈ 2,500 steps × 16 prompts × 16 generations × 4K ctx): 20 steps × 8 prompts ×
8 generations × 1,024 max new tokens, seed 0, same fixed 512-problem DeepScaleR slice.
That is ~1,280 training rollouts against their ~645K — a gain-at-matched-compute probe,
not a full-recipe reproduction.

**Eval:** MATH500 + AMC23, greedy pass@1, the rig's t073-validated pipeline (vLLM
generation + the authors' vendored Qwen2.5-Math grader), identical prompts and grader for
every arm. Steering checkpoints served in **stock vLLM** via an architecture re-badge:
Qwen2.5 relabeled `LlamaForCausalLM` with `mlp_bias=true` (Qwen2.5 is Llama-shaped;
zero-filled o/gate/up biases; learned vectors in the down biases) — logit-equivalence
verified (fp32 max|Δ| = 0.0; bf16 argmax-identical on all probes).

## Results: a five-arm null

| arm | MATH500 | Δ vs base | AMC23 | Δ |
|---|---|---|---|---|
| base | 54.6 | — | 45.0 | — |
| steering @ 20 steps | 54.4 | −0.2 | 45.0 | 0.0 |
| LoRA r4 down_proj @ 20 steps | 53.8 | −0.8 | 40.0 | −5.0 |
| random-reward steering @ 20 steps | 54.2 | −0.4 | 45.0 | 0.0 |
| base + "To" prefix (no training) | 53.0 | −1.6 | — | — |
| *their claim (full recipe)* | *79.9* | *+27.7* | *62.5* | *+16.7* |

Base sanity holds: 54.6 MATH500 / 45.0 AMC23 vs their reported base 52.2 / 45.8 (and the
rig's t073 measurement 53.4) — the pipeline reproduces their starting point. The trained
arms are genuinely different checkpoints (steering vector norms ~0.30–0.32 per layer,
LoRA 2.52M params trained, generations shifted) — they just don't score differently.

Three reads:

1. **The steering gain does not materialize early.** At 20 matched steps the +27.7 claim
   shows +0.0. Whatever their curve does, none of it lives in the first ~1,280 rollouts —
   on a consumer card the recipe's cost-to-first-signal is the whole recipe, not a cheap
   probe. (Contrast One-Shot-EM, same base family, which moved +2.0 by step 10 — t073.)
2. **Correct rewards buy nothing over random ones here.** Random-reward steering (54.2 /
   45.0) is indistinguishable from correct-reward steering (54.4 / 45.0) — both are base.
   At this budget there is no evidence the reward signal, the thing RL is supposedly
   injecting, does anything at all.
3. **Their own token-substitution number does not reproduce on the standard template.**
   "To"-prefix: −1.6, not +10–11. The likely reason is right in the raw generations: on
   the `qwen_math` CoT template the base model's answers *already open with "To"* — the
   token the last-layer vector allegedly boosts is already there. The 10–11-point claim
   must live in a weaker base protocol; on the template their own training config uses,
   there is nothing for first-token substitution to elicit.

## Where the wall-clock actually goes

Measured per step, median across all three trained arms (one RTX 5090): rollout ~50s
(**75–76%**), reward grading ~0.3s (<1%), backward+update ~16s (**24–25%**). The
per-step split is IDENTICAL whether 100K bias params or 2.52M LoRA params train, because
the update slice is dominated by the backward pass through the frozen 7.6B network —
which no adapter scheme shrinks. Their "0.11s vs 9.94s" can only be the optimizer sliver
inside that slice; their "34s vs 52m" headline excludes the ~75% of wall-clock (rollouts)
that is invariant to what trains.

On a 32GB consumer card the honest efficiency win of bias-only adaptation is **memory,
not speed**: full-param 7B RL does not fit at all (t073's OOM gate), while bias-only
(~100K params, ~800KB optimizer state) trains comfortably. That is a real and useful
result — it is just not "34s training."

## Honest limits

- 20 steps is ~0.2% of their rollout budget; this bounds where the gain ISN'T (early),
  not whether their endpoint is real. The Table-1 claim is neither confirmed nor refuted.
- Their full-FT arm cannot run on 32GB; their published number is shown as reference.
- Qwen2.5-14B untouched (bf16 weights alone ≈ 29GB).
- Single seed; AMC23 is 40 questions (±2.5pts/question); LoRA's −5.0 there is 2 questions.
- Our loop uses HF-generate rollouts (their vLLM-engine rollouts are faster in absolute
  terms); the SHARE decomposition, not the absolute seconds, is the finding.

## Artifacts

- `lib/steering.py` + tests (budget chooser, Qwen2→Llama re-badge config, timing parse)
- `scripts/train_steering.py` (bounded RLOO; steering / their-config LoRA / random-reward arms; timing instrument)
- `scripts/convert_steering_checkpoint.py` + `scripts/steering_equiv_check.py` (re-badge + fp32-verified gate)
- `scripts/fetch_steering_data.py`, `scripts/aggregate_steering.py`, `scripts/chart_steering.py`
- `scripts/eval_math.py --assistant-prefix` (the zero-training "To" probe)
- Chart: `reports/steering-claimed-vs-measured.png`
