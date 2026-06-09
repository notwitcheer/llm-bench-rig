# Agentic Score leaderboard (7 models): small models punch up, long-context splits the field

**Rig:** one RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Harness:** 40 deterministic tasks — 36 across **five short-context axes** (chain, multistep, coding,
error-recovery, distractor) → the Agentic Score, plus a **separate long-context axis** (`read_doc` returns
a seeded document with a buried activation code, at 32K and 128K). Programmatic verification; sub-5% = tie.

## The board

| # | model | params | Agentic Score | success | tool-eff | tok/task | error-rec | lc@32K | lc@128K |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.6-27B | 27B | **98.6** | 97% | **1.00** | 285 | 6/6 | 100% | **100%** |
| 2 | Qwen3.5-35B-A3B (base) | 35B-A3B | 97.5 | 100% | 0.88 | 241 | 6/6 | 100% | 50%* |
| 3 | **Qwopus-GLM-18B** | 18B | 97.1 | 97% | 0.92 | 232 | 6/6 | 100% | **100%** |
| 4 | Nemotron-Cascade-2-30B | 30B-A3B | 96.9 | 100% | 0.85 | 320 | 6/6 | 50% | **0%** |
| 5 | Kimi-Linear-48B-A3B | 48B-A3B | 92.9 | 94% | 0.78 | 185 | 6/6 | 100% | **100%** |
| 6 | Granite-4.1-30b | 30B | 92.0 | 86% | 0.95 | **79** | **2/6** | 50% | **OOM** |
| 7 | Nex-N2-mini | 35B-A3B | 90.4 | 83% | 0.94 | 82 | 4/6 | 100% | **100%** |

## Findings

**The top four are a tie.** Qwen3.6-27B (98.6), base Qwen3.5 (97.5), Qwopus-GLM-18B (97.1) and
Nemotron-Cascade-2 (96.9) are within 1.7 points — a statistical wash on the score alone. The real
separation is in **tokens, error-recovery, and long-context**, not the headline number.

**The model the rig runs is the best agent tested.** Qwen3.6-27B — *Donald's own deployed model* — tops
the board with a perfect 1.00 tool-efficiency (never a wasted call) and holds 128K. Nice to learn your
own stack is well-chosen.

**Small models punch up — size isn't the story, training is.** **Qwopus-GLM-18B**, an 18B *community
merge*, lands 3rd: it beats Kimi-Linear-48B, Granite-30B and Nex-35B outright, is the **leanest of the
top cluster** (232 tok), and **holds 128K at 100%** on just 13GB of weights. The 18B punches two weight
classes up.

**Long-context needs two things: room and retrieval.** Four models hold 128K needle-finding at 100%
(Qwen3.6-27B, Qwopus-18B, Kimi-Linear, Nex). Two fail for *different* reasons:
- **Granite-4.1-30b walls (OOM)** — a VRAM problem (its KV won't fit 128K on 32GB), and even at 32K it's
  only 50%.
- **Nemotron-Cascade-2 fits 128K but retrieves 0/2** — an *ability* problem, not memory. The rig's old
  speed-king serves the context fine and then can't find the needle in it.

**Error-recovery isolates one model.** Six of seven recover 5–6/6 when a tool errors with a hint
(`ids are zero-padded`, `use https`). **Granite recovers 2/6** — it's the lean, tight-tool-use model that
falls apart the moment a tool pushes back. Nex is the other soft spot (4/6).

\* Qwen base 128K = 1/2: one request hit a server-side 500 at the context edge (infra, not a model miss);
recorded honestly.

## Worth it?

- **Qwen3.6-27B** — the best all-round agent here, and what the rig already runs.
- **Qwopus-GLM-18B** — the value pick: top-cluster quality, long-context, smallest footprint.
- **Granite-4.1-30b** — cheapest per task (79 tok) but brittle: no error-recovery, no long context. Short,
  well-behaved loops only.
- **Nemotron-Cascade-2** — strong short-task completion, but token-heavy and blind past short context.

Cheap, robust, and long-context-capable are three different axes. The hardened suite finally measures all
three — and no single model wins all of them.

---

*Harness: `lib/agentic/native/` in notwitcheer/llm-bench-rig (34 unit tests). Charts: frontier / score /
long-ctx. Live board: witcheer/agentic-score-leaderboard.*
