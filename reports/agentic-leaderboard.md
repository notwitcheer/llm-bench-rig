# Agentic Score leaderboard: Granite-4.1-30b owns the efficiency frontier

**Rig:** one RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Harness:** the rig's Agentic Score — a model-agnostic native tool-calling loop over 15 deterministic,
programmatically-checked tasks (tool-use chains, multi-step dependencies, sandboxed coding).
Calibration-grade; a sub-5% gap is a tie.

## The board

| model | params | Agentic Score | task success | tool-eff | tokens/task | gate |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (base) | 35B-A3B | **98.0** | **100%** (15/15) | 0.90 | 264 | ✓ |
| **Granite-4.1-30b** | 30B | 96.0 | 93% (14/15) | **0.97** | **75** | ✓ |
| Nex-N2-mini | 35B-A3B | 92.0 | 87% (13/15) | 0.93 | 93 | ✓ |
| Kimi-Linear-48B-A3B | 48B-A3B | 89.9 | 93% (14/15) | 0.66 | 252 | ✓ |

All four pass the native tool-calling gate, and all four nail the multi-step axis (5/5) — the spread is
in chains, coding, and efficiency.

## The find: success and token-efficiency are *separable* axes

The Agentic Score rewards completion, but the **efficiency frontier** (success vs tokens/task) is where
the models separate:

- **Granite-4.1-30b is the quiet winner.** A 30B that nobody calls an "agent model" does near-base work
  (93% vs 100%) at **a third of the tokens** (75 vs 264) with the tightest tool use on the board (0.97 —
  almost no wasted calls). On one 5090 it's the cheapest competent agent here, by a wide margin.
- **Base Qwen3.5-35B still tops raw completion** (100%) but pays for it in tokens (264).
- **Nex-N2-mini** is the lean post-train (93 tok) but lowest completion (87%) — its Adaptive Thinking
  trades success for brevity (see the Nex report).
- **Kimi-Linear-48B-A3B** matches Granite's success (93%) but spends **3.3x the tokens** (252) and
  over-calls tools (0.66 tool-eff — it routinely double-calls where one call suffices).

## Two notes worth their own line

**Kimi-Linear runs natively on one 5090 — a feasibility win.** Its `kimi_linear` arch (Kimi Delta
linear-attention, 256-expert A3B MoE) is compiled into llama.cpp b9562 and loaded at full `-ngl 999`
offload without OOM — the linear-attention KV cache is small enough that the 28GB Q4 fit the 32GB card
with room to spare. But its architectural edge is **long context**, which this short-task bench does not
exercise; the verbosity and over-calling are real agentic-economy observations, not a verdict on the
model's reason for existing. A long-context agentic test is the fair follow-up.

**Granite's only miss was a Fibonacci off-by-one** (`coding_fib10`) — the classic indexing slip, not a
tool-use failure. Kimi's only miss was over-thinking a trivial halving (483 tokens to fail
`chain_config_half`) — a brevity problem, not a capability gap.

## Worth it?

- **Granite-4.1-30b** if you want a cheap-to-serve agent: near-base completion at a third of the tokens
  and the tightest tool loop here. The standout of this round.
- **Base Qwen3.5-35B** if maximum completion beats token budget.
- **Kimi-Linear** if your workload is genuinely long-context (its real strength, untested here) — not for
  short, token-sensitive agentic loops.

---

*Harness: `lib/agentic/native/` in notwitcheer/llm-bench-rig (17 unit tests). Charts:
`reports/agentic-leaderboard-frontier.png`, `reports/agentic-leaderboard-score.png`.*
