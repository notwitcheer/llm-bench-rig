# Agentic Score leaderboard (hardened): efficiency and robustness pull apart

**Rig:** one RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Harness (hardened):** 40 deterministic tasks — 36 across **five short-context axes** (chain, multistep,
coding, **error-recovery**, **distractor**) scored into the Agentic Score, plus a **separate long-context
axis** (a `read_doc` tool returns a seeded document with a buried activation code, at 32K and 128K).
Programmatic verification; a sub-5% score gap is a tie.

## The board

| # | model | params | Agentic Score | success | tool-eff | tok/task | error-rec | lc@32K | lc@128K |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.5-35B-A3B (base) | 35B-A3B | **97.5** | 100% | 0.88 | 241 | 6/6 | 100% | 50%* |
| 2 | **Kimi-Linear-48B-A3B** | 48B-A3B | 92.9 | 94% | 0.78 | 185 | 6/6 | 100% | **100%** |
| 3 | Granite-4.1-30b | 30B | 92.0 | 86% | **0.95** | **79** | **2/6** | 50% | **OOM** |
| 4 | Nex-N2-mini | 35B-A3B | 90.4 | 83% | 0.94 | 82 | 4/6 | 100% | **100%** |

## What hardening revealed (the thin 15-task board hid all of this)

**1. The board reshuffled.** Granite-4.1-30b — the previous "efficiency king" at 96 on the easy suite —
dropped to 92.0 and 3rd. Kimi-Linear rose. A 15-task board that everyone nearly aced wasn't measuring
the things that separate agentic models.

**2. Error-recovery is the discriminating axis.** When a tool returns an error **with a hint**
(`record '7' not found; ids are zero-padded`, or `use https://`), the model has to adapt. Base Qwen and
Kimi recover cleanly (6/6) — which proves the tasks are solvable — but **Granite recovers only 2/6**. It
is lean and tight on the happy path and then doesn't adjust when a tool pushes back. This is exactly the
failure mode single-call function-calling benchmarks (BFCL etc.) never test, and it's most of why Granite
fell. Nex is middling (4/6).

**3. Long-context reach splits on architecture.** The A3B models — Kimi-Linear (linear attention) and
Nex-N2-mini (standard A3B MoE) — both hold **100% needle-finding at 128K** on one 5090; their modest KV
fits alongside the weights. The **dense Granite-4.1-30b walls at 128K** (OOM) and only finds the needle
50% of the time even at 32K — the compact model is the weakest at long context. Kimi-Linear is the
cleanest of all: 100% at both tiers, and it's the largest model on the board (48B).

\* **Qwen base at 128K = 50%** with an asterisk: one of its two 128K requests returned a server-side
**500** at the very edge of the context window (an llama.cpp hiccup, not a clean model miss); the other
succeeded. Recorded honestly as 1/2. (The harness now guards per-task exceptions so one bad request can't
void a tier — a fix this run forced.)

## No single winner

- **Base Qwen3.5-35B** tops the Agentic Score (100% short-task success) — but it's the token glutton (241)
  and shaky at 128K.
- **Kimi-Linear-48B** is the best all-rounder: 2nd on score, **and the long-context champion** (100% at
  128K). If your agent loops over big contexts, this is the pick.
- **Granite-4.1-30b** is the cheapest competent agent on the happy path (79 tok/task, 0.95 tool-eff) but
  **brittle**: it can't recover from tool errors and can't hold long context. Great when nothing goes
  wrong; risky when something does.
- **Nex-N2-mini** is lean and reaches 128K, but lowest short-task success.

## Worth it?

Cheap and robust are different axes, and this suite finally measures both. Reach for **Kimi-Linear** for
long-context agentic work, **base Qwen** when you need maximum completion and can pay the tokens, and
**Granite** only for short, well-behaved tool loops where its efficiency shines and nothing will push back.

---

*Harness: `lib/agentic/native/` in notwitcheer/llm-bench-rig (34 unit tests). Charts:
`agentic-leaderboard-frontier.png`, `-score.png`, `-longctx.png`. Live board:
witcheer/agentic-score-leaderboard.*
