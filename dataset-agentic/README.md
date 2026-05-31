---
license: mit
task_categories:
  - text-generation
tags:
  - agent
  - agentic
  - hermes
  - codeact
  - tool-use
  - local-llm
  - gguf
  - rtx-5090
  - benchmark
pretty_name: Hermes Pairing — Agentic Benchmark for Local LLMs
size_categories:
  - n<1K
configs:
  - config_name: phase_a_synthetic
    data_files: hermes_pairing.csv
  - config_name: phase_b_real_harness
    data_files: hermes_pairing_phaseb.csv
---

# Hermes Pairing — Agentic Benchmark for Local LLMs (Phase A)

How well does a local LLM **drive an agent**? This dataset holds results for pairing local models with
[Hermes Agent](https://hermes-agent.nousresearch.com) (NousResearch) — a **CodeAct** agent: the model
acts by writing Python (`execute_code`) that orchestrates tools, *not* by emitting JSON function calls.

Generated with [llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig) on an NVIDIA RTX 5090 (32GB),
llama.cpp / GGUF, **under Hermes's real ~3.5K-token system prompt**.

> **Phase A (synthetic).** A reproducible synthetic ranking of agentic *capability* — not a real-harness
> verdict. Phase B (running the top finishers through the actual Hermes Agent) is the validation step.

## Leaderboard

| Rank | Model | Pairing | codeact | multistep | instruction | long-ctx |
|---|---|---|---|---|---|---|
| 1 | Qwopus-GLM-18B (13GB) | **92.65** | 94 | 75 | 94 | 100 |
| 2 | Qwen3.6-27B (21GB) | **92.38** | 99 | 100 | 100 | 71 |
| 3 | Nemotron-Cascade-2-30B | 90.50 | 99 | 50 | 92 | 100 |
| 4 | Hermes-4.3-36B (21GB) | 84.27 | 95 | 67 | 98 | 67¹ |

¹ Hermes-4.3-36B OOMs at 64K context on 32GB → 0 at that depth.

## The four axes (Hermes Pairing Score = weighted sum, 0–100)

- **codeact** (0.40) — code-as-action: writes Python orchestrating tools, executed in a sandbox (pass@1).
- **longcontext** (0.25) — retrieve-and-use a needle in a long memory/tool-log context at 16K/32K/64K (per-depth, VRAM-aware; an OOM depth scores 0).
- **instruction** (0.20) — compliance under the real Hermes prompt vs minimal (the delta is the heavy-prompt tax).
- **multistep** (0.15) — loop stability: forced fetch→observe→act loops.

Per-depth long-context and the instruction delta are in `hermes_pairing.csv`.

## Findings (honest)

- **#1 and #2 are a statistical tie** (within ~±5% CI). Not "Qwopus won by 0.3."
- **An 18B frankenmerge competes with the bigger models** — efficiency over raw size.
- **The lab's own model finishes last** — Hermes-4.3-36B is the worst pairing for Hermes Agent (gap is real).
- **No model wins all four axes** — best agent model depends on workload: Qwen 27B is the reasoning-loop/instruction king but weak at retrieval; Nemotron + Qwopus ace long-context; Nemotron is weakest at multi-step.
- **The 64K VRAM wall** — a 36B Q4 can't hold 64K KV in 32GB; smaller models can. Size is a liability for long-context agents on a single card.

## Phase B — real-harness validation (`phase_b_real_harness`)

The synthetic Phase A was confirmed/broken by running the top 3 finishers through the **real Hermes
Agent** (no-sudo `uv` install, one-shot mode, pointed at llama-server) on **14 multi-step tasks × 3
repeats = 126 runs**. Each task succeeds only if the resulting filesystem state is correct. Efficiency
(turns = model calls; gen tokens) parsed from the inference-server log.

| Model | Completion | Avg turns | Avg gen tokens |
|---|---|---|---|
| **Qwen3.6-27B** | **100%** (42/42) | **3.0** | **364** |
| Qwopus-GLM-18B | 85.7% (36/42) | 3.6 | 870 (2.4×) |
| Nemotron-Cascade-2-30B | 85.7% (36/42) | 4.4 | 1334 (3.7×) |

**The synthetic tie broke.** Qwen3.6-27B completed every task *and* was 2.4–3.7× more token-efficient —
an efficiency gap a synthetic benchmark can't measure (visible only inside the real agent loop). The
18B that tied #1 synthetically is neither the most reliable nor the most efficient in the real harness.
Qwopus and Nemotron fail on *different* task types (parsing/logic vs. transforms/chains). **Overall
verdict across both phases: Qwen3.6-27B.**

## Method

llama.cpp / `llama-server`, GGUF, RTX 5090 32GB. Real Hermes prompt extracted from the open-source repo
(static guidance, ~3.5K tokens). `codeact` runs under a light prompt (the real prompt conditions models
into incremental tool-call markup that fights a one-shot block eval); the other three run under the real
prompt. Decision-run sizes: codeact n≈100, instruction n≈50, longcontext n≈15×3 depths, multistep n≈12.
Full code + methodology: https://github.com/notwitcheer/llm-bench-rig (`lib/agentic/`).

## Models
[KyleHessling1/Qwopus-GLM-18B-Healed](https://huggingface.co/KyleHessling1/Qwopus-GLM-18B-Healed) ·
[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) ·
[bartowski/Nemotron-Cascade-2-30B-A3B-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF) ·
[NousResearch/Hermes-4.3-36B-GGUF](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF)
