# Qwable-v1: a Claude-Code-distilled "agentic coder" that's worse than its own base — measured

**Rig:** one RTX 5090 32GB · llama.cpp b9653 (qwen35moe) · llama-server `--jinja` native tool-calling · Q5_K_M for all three models (no quant confound)
**Model:** [lordx64/Qwable-v1](https://huggingface.co/lordx64/Qwable-v1) — Qwen3.6-35B-A3B (MoE, 3B active) → SFT distill of **Claude Opus-4.7 reasoning** → SFT on **Claude Fable-5 agentic tool-use traces**. A Claude-Code-style agentic coder (emits `<think>` then `<tool_call>`). Brand-new, no published evals. The "agentic SFT → better agent" premise is the contestable claim.
**Setup:** the rig's native Agentic Score (40 tool-calling tasks, 5 axes) + the SWE-bench Verified reality anchor (30-bug subset, official harness in Docker). Run as a **controlled 3-way**: vanilla base → +Opus-distill → +Fable-5 SFT (Qwable), same quant, same harness — so any delta is the post-training, not the model family or quant.

## The numbers

| pipeline stage | Agentic Score | SWE-bench Verified (30) | empty patches |
|---|---|---|---|
| Qwen3.6-35B-A3B (vanilla base) | **99.58** (board #2) | **19/30 (63%)** | 9 |
| + Opus-4.7 reasoning distill | 97.92 (#4) | — | — |
| + Fable-5 agentic SFT = **Qwable-v1** | **96.25** (#8) | **11/30 (37%)** | 16 |

**Every post-training step made it worse.** The synthetic agentic score declines monotonically (99.58 → 97.92 → 96.25), and on real bugs the finished model resolves **8 fewer** than its untuned base (19 → 11) while **giving up nearly twice as often** (9 → 16 empty patches).

## What it means

- **The vanilla base is excellent.** Qwen3.6-35B-A3B lands #2 on the agentic board and **ties the board's best real SWE-bench resolve (19/30)** — as good as the dense Qwen3.6-27B. The "agentic coder" distillation took a top-tier base and regressed it.
- **Not a mirage — a regression.** Qwable's synthetic score (96.25, #8) *fairly predicts* its real rank (11/30, mid-low), unlike the in-distribution inflation seen in models trained on the bench's own dialect (e.g. a sibling coder that posted a perfect synthetic 100 then resolved below its base). Here the synthetic number is honest; the model genuinely got worse.
- **Why distilling agentic traces can cost capability:** ~12M tokens of one developer's Claude-Code sessions (2-epoch LoRA) narrows the policy toward a specific trace style. The give-up rate nearly doubling (9 → 16 empty patches) is the tell — the SFT taught it to *emit fewer, more cautious* trajectories, not to *solve more*. Reasoning distillation (the middle step) already costs 1.66 points before the agentic SFT costs another 1.67.

## Honest caveats

- Q5_K_M throughout (no quant confound), but a single seed per model on a 30-bug subset — small-n; the *direction* (base > distill > Qwable on both axes) is the robust signal, not the exact bug counts.
- The Opus-distill's SWE-bench was not run (only its agentic score); the real-bug controlled pair is base vs Qwable.
- This measures *this* distillation recipe (one dev's traces, 2-epoch LoRA), not "agentic distillation" in general.

## Reproduce

`gate_and_run.sh <gguf> <slug>` (native Agentic Score, Donald-safe) + `swe_gen_one.sh`/`swe_grade_one.sh` (SWE-bench gen + official grade, 30-bug `swebench_ids_30.txt`) + `build_agentic_leaderboard.py` (META line + re-run). Three models, ~5h on one 5090.
