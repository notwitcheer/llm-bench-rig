# Hermes Pairing — agentic benchmark for local LLMs

A benchmark for **how well a local LLM drives an agent** — specifically [Hermes Agent](https://hermes-agent.nousresearch.com) (NousResearch), a **CodeAct** agent: the model acts by writing Python (`execute_code`) that orchestrates tools, *not* by emitting one JSON function call at a time.

Most "function-calling" benchmarks score JSON tool calls. That's the wrong axis for a code-as-action agent. This suite measures what actually matters for pairing a model with such an agent, **under the agent's real system prompt**.

> **Status: Phase A (synthetic).** Results are a reproducible synthetic ranking — not a real-harness verdict. Phase B (running the top finishers through the actual Hermes Agent) is the validation step. Treat Phase A as "which models are *capable* of the agentic primitives," not "which model is best in production."

## The four axes

| Eval | Weight | Measures |
|---|---|---|
| `codeact` | 0.40 | Code-as-action: model writes Python that orchestrates mock tools to complete a task (pass@1, executed in a sandbox). Run under a **light** prompt — see note. |
| `longcontext` | 0.25 | Retrieve-and-use a needle buried in a long simulated memory/tool-log context, at 16K / 32K / 64K. Per-depth VRAM-aware (a depth that OOMs scores 0). |
| `instruction` | 0.20 | Instruction-following compliance **under the real ~3.5K-token Hermes prompt** vs a minimal one (reports the delta — the capability tax of the heavy prompt). |
| `multistep` | 0.15 | Loop stability: forced fetch→observe→act loops where the goal requires a value only obtainable from a prior tool observation. |

**Hermes Pairing Score** = weighted sum of the four (0–100).

## Design notes (the honest part)

- **Real prompt, extracted, not invented.** `scripts/extract_hermes_prompt.py` pulls Hermes's static stable-tier guidance (identity + tool-use/`execute_code` enforcement + memory/skills guidance) from the open-source repo via `ast` — no Hermes dependencies. ~3.5K tokens. The full runtime prompt is larger (dynamic tool schemas + skills index); we use the static behavioral stressor.
- **Why `codeact` uses a *light* prompt.** Under Hermes's real prompt, models adopt Hermes's *incremental* tool-call style (`<tool_parsing>` markup, one action then await observation) — which fights a one-shot "write a self-contained block" eval (codeact collapsed 5/5→1/5 in calibration). So `codeact` measures pure code-orchestration capability under a light prompt; the real incremental behavior is what Phase B measures. `instruction`/`longcontext`/`multistep` run under the real prompt (they don't fight it).
- **Mock tools, deterministic.** All tool outputs are seeded and reproducible (`lib/agentic/mock_tools.py`), so scoring is deterministic and the benchmark is re-runnable.
- **Sizes & CI.** Decision run: codeact n≈100, instruction n≈50, longcontext n≈15×3 depths, multistep n≈12. Top results within ~±5% CI — **a 0.3-point gap is a tie, not a winner.**

## Layout

```
lib/agentic/
  mock_tools.py     deterministic mock tools
  sandbox.py        subprocess executor for code actions
  base.py           prompt loader, code extraction, tool doc
  codeact.py        eval 1 (+ check_result)
  instruction.py    eval 2 (compliance under load)
  longcontext.py    eval 3 (needle-in-haystack, per-depth)
  multistep.py      eval 4 (loop stability)
  score.py          Hermes Pairing Score aggregation
  run.py            orchestration (two-phase server, per-depth longcontext)
data/agentic/*.jsonl   the eval datasets
scripts/extract_hermes_prompt.py   real-prompt extractor
```

## Run

```bash
# 1. extract the real Hermes prompt (after cloning hermes-agent)
python3 scripts/extract_hermes_prompt.py ~/hermes-src

# 2. set agentic.models + depths in config.yaml, then
./run_agentic.sh    # or: python -m lib.agentic.run
# -> results/<slug>/agentic.json + results/hermes_pairing_leaderboard.json
```

Requires a local OpenAI-compatible endpoint (llama.cpp `llama-server`) on `:8090`.

## Results (Phase A, 4 models, RTX 5090 32GB)

| Rank | Model | Pairing | codeact | multistep | instruction | long-ctx |
|---|---|---|---|---|---|---|
| 1 | Qwopus-GLM-18B | 92.65 | 94 | 75 | 94 | 100 |
| 2 | Qwen3.6-27B | 92.38 | 99 | 100 | 100 | 71 |
| 3 | Nemotron-Cascade-2-30B | 90.50 | 99 | 50 | 92 | 100 |
| 4 | Hermes-4.3-36B | 84.27 | 95 | 67 | 98 | 67¹ |

¹ Hermes-4.3-36B OOMs at 64K context on 32GB (KV cache won't fit) → 0 at that depth.

**#1 and #2 are a statistical tie.** The durable signals: an 18B competes with the bigger models, the 36B finishes last, and no model wins all four axes — the best agent model depends on your workload. Real-harness validation (Phase B) is the tie-breaker.
