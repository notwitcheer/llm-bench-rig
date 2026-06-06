# RL on a 5090: FP8 RL hit a triple wall, so portable GRPO shipped a real +7.66

**Goal:** the marquee fine-tuning frontier — **FP8 reinforcement learning** (GRPO) on Blackwell, the one path most likely to hit the same toolchain wall that blocked vLLM inference (t036).
**Model:** `unsloth/Qwen3-4B` + QLoRA GRPO adapter (r=32) · **Reward:** correctness (answer match) + format (`#### <answer>` marker)
**Hardware:** RTX 5090 32GB · `~/grpo-env` (Unsloth 2026.6.1, torch 2.10+cu128, sm_120) · Donald drained for the run, restored after
**Eval:** in-process GSM8K, N=300, greedy, identical prompt + extraction for base and tuned

## Result

| | GSM8K (n=300) |
|---|--:|
| Base (Qwen3-4B) | **60.67%** (182/300) |
| + QLoRA GRPO (correctness + format reward) | **68.33%** (205/300) |
| **Gain** | **+7.66 pts** (+23 questions) |

GRPO mean reward rose **1.375 → 1.725** over 150 steps (num_generations=8) — still climbing, not saturated. A clean, *measured* RL lift on a 4B, bigger than last week's +4.67 SFT gain on a 1B. But the headline isn't the number — it's which door was locked.

## The marquee path was FP8 RL. It hit a triple wall.

The plan was FP8 RL: Unsloth exposes it as `FastLanguageModel.from_pretrained(..., load_in_fp8=True)`, routing rollouts through vLLM for ~1.4x faster RL inference. On Blackwell, that path is blocked three independent ways — discovered in order:

1. **The t036 signature, again — flashinfer FP8 kernels need CUDA ≥ 12.9.** `load_in_fp8=True` loads `unsloth/Qwen3-4B-FP8` and inits vLLM with `quantization=fp8`. vLLM logs `SM 12.x requires CUDA >= 12.9`; capsule's toolkit is 12.8 (nvcc 12.8.61), so flashinfer can't build the FP8 GEMM kernels. The exact wall t036's quant *inference* hit — now extended to RL.

2. **An unsloth ↔ vLLM LoRA-API version crash.** FP8 RL needs vLLM `fast_inference` + `enable_lora=True`. The only Blackwell-wheel vLLM (0.21.0) forces an *older* unsloth (2026.3.11), and that pair dies at LoRA warmup: `'LRUCacheWorkerLoRAManager' object has no attribute 'get_dummy_lora_warmup_rank'` — an internal version split (unsloth 2026.3.11 + unsloth_zoo 2026.6.1).

3. **An irreconcilable torch pin.** Proven unsloth (2026.6.1) requires torch 2.10; vLLM 0.21 requires torch 2.11+cu130. They cannot share one env. And the unsloth version vLLM *does* allow (2026.3.11) has a broken GRPO loss path (`NameError: align_completion_tool_mask`).

**Verdict:** FP8 RL on this box is blocked by *both* a toolkit limit (CUDA 12.8 < 12.9) *and* a torch/unsloth/vLLM version knot. Same call as t036 — deferred, no toolkit install for one comparison.

## What shipped instead: portable GRPO (use_vllm=False)

Rather than chase a CUDA toolkit, GRPO ran on plain HF generation — vLLM-free, so it sidesteps all three walls at once. Qwen3-4B 4-bit QLoRA policy, two rewards (correctness via the same GSM8K extractor the eval uses, plus a `####`-format reward), the **same prompt as the eval** (train↔eval alignment — last week's hardest-won lever). ~18.8s/step at num_gen=4; the full run used num_gen=8, 150 steps. Slower than a vLLM rollout would be — but it produces the gain on hardware where the fast path won't load.

## Worth it if / not if

- **The portable path is the pragmatic one.** If your box lacks a CUDA toolkit (or you just don't want the rabbit hole), `use_vllm=False` GRPO trades rollout speed for a stack that actually runs. The +7.66 is real either way.
- **FP8 RL needs CUDA ≥ 12.9 on Blackwell.** Not the model's fault, not Unsloth's — it's flashinfer's kernel build. Plan for a toolkit install if FP8 rollouts are the point.
- **Not a capability miracle.** GRPO on a 4B with a correctness+format reward sharpens what the model can already nearly do (reward climbed, didn't explode). A few points of measured GSM8K, not a tier jump.

## Honest scope

In-process eval (internally consistent: base vs tuned, identical harness, raw generations saved for offline re-scoring) — not the leaderboard's server harness, so the absolute % isn't directly comparable to the model board. The **delta** is the claim.

## Reproduce

```bash
# portable GRPO (no vLLM, no toolkit) — Qwen3-4B 4-bit QLoRA
~/grpo-env/bin/python scripts/train/gsm8k_grpo.py            # -> ~/unsloth-runs/gsm8k-grpo/adapter
~/grpo-env/bin/python scripts/train/eval_gsm8k.py --base unsloth/Qwen3-4B --out eval_base.json
~/grpo-env/bin/python scripts/train/eval_gsm8k.py --base unsloth/Qwen3-4B \
    --adapter ~/unsloth-runs/gsm8k-grpo/adapter --out eval_tuned.json
```
Scripts: `scripts/train/{gsm8k_grpo,grpo_rewards,eval_gsm8k,gsm8k_extract}.py`. Adapter: `witcheer/qwen3-4b-gsm8k-grpo`.
