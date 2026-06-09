---
license: mit
tags:
- agentic
- tool-calling
- llm-leaderboard
- benchmark
- rtx-5090
pretty_name: Agentic Score Leaderboard (one RTX 5090)
configs:
- config_name: default
  data_files: leaderboard.csv
---

# 🛠️ Agentic Score Leaderboard — one RTX 5090

How well do local models actually **drive a tool-using agent loop**? Not single-call function-calling
benchmarks — a real loop: native OpenAI tool-calling through `llama-server`, multi-step deterministic
tasks, programmatic verification. Everything runs on a **single RTX 5090 32GB**.

Updated **2026-06-09** · llama.cpp b9562 · `--jinja` native tool-calling · temp 0.

## Leaderboard

| # | model | params | Agentic Score | success | tool-eff | tokens/task | chain | multistep | coding | lc@32k | lc@128k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.5-35B-A3B (base)** 🏆 | 35B-A3B | **97.5** | 100% | 0.88 | 241 | 8/8 | 8/8 | 8/8 | 100% | 50% |
| 2 | **Kimi-Linear-48B-A3B** | 48B-A3B | **92.91** | 94% | 0.78 | 185 | 7/8 | 7/8 | 8/8 | 100% | 100% |
| 3 | **Granite-4.1-30b** | 30B | **92.04** | 86% | 0.95 | 79 | 8/8 | 8/8 | 7/8 | 50% | OOM |
| 4 | **Nex-N2-mini** | 35B-A3B | **90.42** | 83% | 0.94 | 82 | 7/8 | 7/8 | 7/8 | 100% | 100% |

*tool-eff = tool calls vs optimal (1.0 = no wasted calls). tokens/task = avg completion tokens (lower =
leaner). A sub-5% score gap is a tie.*

![efficiency frontier](agentic-leaderboard-frontier.png)

![agentic score](agentic-leaderboard-score.png)

![long-context reach](agentic-leaderboard-longctx.png)

## The Agentic Score (0–100)

Aggregate over 36 deterministic short-context tasks across five axes (tool-use chains, multi-step
dependencies, sandboxed coding, **error-recovery**, **distractor-robustness**), weighted as below.
A separate **long-context** axis (needle-in-a-document at 32K / 128K) is reported in the `lc@` columns,
not blended into the score (so a 128K VRAM wall doesn't corrupt it):

| axis | weight | measures |
|---|---|---|
| Task success | 0.50 | % tasks completed correctly (programmatic check) |
| Tool efficiency | 0.20 | tool calls vs optimal; malformed/wasted calls penalized |
| Token efficiency | 0.15 | avg tokens/task (efficiency at equal success) |
| Loop stability | 0.15 | completes without stalling / exceeding the step cap |

Calibration-grade (synthetic, deterministic, re-runnable) — a reality anchor (SWE-bench-style) is future
work. Harness + 17 unit tests: **[notwitcheer/llm-bench-rig](https://github.com/notwitcheer/llm-bench-rig)**
(`lib/agentic/native/`).

## Notes per model

- **Qwen3.5-35B-A3B (base)** (bartowski/Qwen_Qwen3.5-35B-A3B-GGUF): generalist base, no agentic post-train
- **Kimi-Linear-48B-A3B** (bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF): linear-attention; runs natively on one 5090; long-context untested here
- **Granite-4.1-30b** (unsloth/granite-4.1-30b-GGUF): leanest competent agent on the board
- **Nex-N2-mini** (eramax/Nex-N2-mini-gguf): agentic post-train of Qwen3.5; Adaptive Thinking

## How it grows

Each model: served Donald-safe on the 5090, hard-gated for parseable native `tool_calls`, then run
through the harness. New models are appended as they're benched. Companion write-ups live in the
[rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks) dataset.
