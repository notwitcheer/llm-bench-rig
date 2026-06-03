---
type: findings
topic: 5090-benchmarks
created: 2026-06-03
updated: 2026-06-03
authoritative_source: ~/benchmark-rig/reports/*.md + HF witcheer/rtx-5090-benchmarks
---

# 5090 benchmark findings (OUR measured results — ground all benchmark claims here)

When asked about any of these, use THESE numbers. Do not substitute papers or training memory.
All runs: one RTX 5090 (32GB), reasoning ON, temperature 0, ~100 items/task (MMLU 114 = 2/subject).
Authoritative detail: `~/benchmark-rig/reports/<slug>.md`.

## gpt-oss-120B (OpenAI, 117B-A5.1B, native MXFP4, 59GB) — runs via MoE CPU-offload
`--n-cpu-moe 20`, ~30GB of 32GB VRAM + rest in system RAM. slug: gpt-oss-120b-mxfp4.
- MMLU 89.5 | ARC-C 95.0 | GSM8K 97.0 | HellaSwag 80.0 | HumanEval 98.0
- generation 47 tok/s | prefill pp512 473 tok/s
- finding: a frontier 120B runs on one consumer card via offload; gen is slow (47 tok/s) because the
  experts live in RAM (CPU-bandwidth bound); only 5.1B active is what makes it feasible. VRAM decides
  if it LOADS, active params + RAM bandwidth decide how fast it RUNS.

## Qwen3.6-28B-REAP20-A3B (20% expert-pruned from the 35B, Q6_K, full VRAM, 21.6GB)
slug: qwen3-6-28b-reap20-a3b-q6-k.
- MMLU 87.7 | ARC-C 95.0 | GSM8K 90.0 | HellaSwag 82.0 | HumanEval 94.0
- generation 247 tok/s
- finding (OUR head-to-head vs the parent below): REAP pruning is a VRAM play, not a speed play.
  ~6GB VRAM saved (27.3 -> 21.6), NO speed gain (both ~3B active, 247 vs 260 tok/s — REAP slightly
  slower even), and behind the parent on all 5 tasks (modest, n~100). worth it if memory-bound, not for tok/s.
  (Distinct from the REAP *paper*; these are OUR numbers on OUR box.)

## Qwen3.6-35B-A3B (the unpruned parent, UD-Q6_K, full VRAM, 27.3GB)
slug: qwen3-6-35b-a3b-ud-q6-k. Matched conditions for the REAP comparison.
- MMLU 94.7 | ARC-C 97.0 | GSM8K 92.0 | HellaSwag 87.0 | HumanEval 98.0
- generation 260 tok/s

## HumanEval harness bug (methodology — 2026-06-03)
Our generative HumanEval harness was understating reasoning models. 3 bugs: client `.strip()` killed
code indentation; `--n-cpu-moe` shifted the llama-bench parse column; stop sequences (`\ndef`) truncated
inline-reasoning models before the code emitted. After fixing: gpt-oss-120B 22 -> 98, REAP 46 -> 94.
The previously-published "Q2 quant kills code gen — Coder-Next 10.4% HumanEval" claim is FALSE — it was
the harness, not the quant. Public correction posted. Old Qwen-family HumanEval (Coder-Next 10, Qwen27
19, Qwen35 37) are all harness artifacts — withheld on HF pending re-run.
