# HRM-Text-1B: the recurrent 1B holds up on GSM8K — and one missing tensor silently costs 26 points

**Rig:** one RTX 5090 32GB · transformers 5.11 (bf16, SDPA) · greedy, temp 0
**Model:** [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) (Apache 2.0, 1.18B, bf16 safetensors) — the Hierarchical Reasoning Model line ([arXiv 2605.20613](https://arxiv.org/abs/2605.20613)) scaled from the 27M ARC-AGI puzzle solver to a text base model. #1 HF text-gen trending the week of 2026-06-10, with no published decode-speed numbers anywhere. These appear to be the first.

**Setup:** GSM8K test, 200 questions (seed 42), generative + greedy, answers extracted with the rig's
first-committed-answer rule, raw generations captured for offline re-scoring. The model's documented
protocol: prompt wrapped as `<|im_start|>{condition}{question}<|im_end|>`, condition tokens
`<|quad_end|><|object_ref_end|>` (synth+cot), and `token_type_ids=1` over the whole prompt — that last
tensor marks the prompt as a bidirectional PrefixLM prefix. Run twice: once correct, once the way every
standard harness would run it (no `token_type_ids`).

## The numbers

| protocol | GSM8K (n=200, 95% CI) | decode | peak VRAM |
|---|---|---|---|
| paper claim | 84.5 (protocol n/a) | — | — |
| **correct (prefixlm mask)** | **79.5% ± 5.6** | 42.9 tok/s | 2.41 GB |
| default harness (no mask) | 53.5% ± 6.9 | 42.9 tok/s | 2.41 GB |

**The claim holds.** 79.5% at n=200 puts the paper's 84.5 just inside the confidence interval — a slight
shortfall, not a refutation, and protocol differences (their condition tag, extraction rule) could absorb
most of it. A 1B *base* model with a ~$1,500 training budget scoring ~80% generative GSM8K is the real
story; it also answers the bat-and-ball trap correctly in 5 tokens in `direct` mode.

## The trap: −26 points from one missing tensor

HRM-Text is trained with PrefixLM masking: the prompt attends bidirectionally, generation is causal.
At inference that mask is requested via `token_type_ids=1` — a tensor **no standard eval harness passes**
(lm-eval-harness, lighthouse-style loglikelihood runners, naive `generate()` wrappers all omit it).
Omitting it doesn't error. The model just runs causal-only and quietly loses **26 points** (79.5 → 53.5).

Two practical consequences:

- **Community numbers for this model will disagree wildly**, and the low ones will be measuring their
  harness, not the model. (The rig's oldest lesson — a capable model scoring absurdly low is almost
  always the harness — now has its cleanest quantified example.)
- Condition tokens matter too: in smoke tests, `synth+cot` produced correct Rayleigh-scattering physics
  where `direct`/`cot` confabulated on the same prompt. The tags select training-data distributions, not
  output formats.

## The recurrence bill

The architecture runs two 16-layer stacks in a nested recurrence (2 H-cycles × 3 L-cycles + 1 update
per H — 128 layer invocations per forward, "latent reasoning" instead of emitted CoT). Measured costs:

- **42.9 tok/s decode** (bf16, batch 1, dead-stable across runs) — a 1.2B that decodes like a ~5B dense.
- **128 KV-cache slots, 0.88 MB/token measured** — 4× the cache of a normal 32-layer model (~3.6 GB at
  the full 4K context).
- Weights are only 2.2 GB and peak VRAM 2.41 GB — it ran alongside a live 27B llama-server the whole time.
- **No llama.cpp path** (arch unsupported upstream, [discussion #23415](https://github.com/ggml-org/llama.cpp/discussions/23415)),
  so no GGUF quant rescue: transformers is the deployment story today, and these are the speeds.

## Honest caveats

- **n=200 sample**, ±5.6 pts CI — fine for verify-or-refute, not for leaderboard precision.
- **The hierarchy attribution is untested here.** The authors dropped ACT/halting (the component the
  ARC Prize post-hoc analysis found most load-bearing in the 2025 HRM), and the paper's own
  FLOPs-matched ablations credit the task-completion training objective for the first chunk of the
  gains, before PrefixLM and the HRM recurrence add theirs. What's verified is the *package*, not *why*.
- Context is 4096 — this is a proof-of-concept base model, not a daily driver.

## Reproduce

`scripts/hrm_probe.py` (smoke / decode bench / KV probe / mask ablation) ·
`scripts/hrm_gsm8k.py --n 200 [--no-prefix-mask]` · raw generations:
`raw/hrm-gsm8k-synth-cot-n200.jsonl`, `raw/hrm-gsm8k-synth-cot-nomask-n200.jsonl` (this dataset).
