# Qwen-AgentWorld's zero-fine-tune transfer doesn't show up on a 5090: flat on synthetic, down on real coding

**Rig:** one RTX 5090 32GB (sm_120) · Qwen-AgentWorld-35B-A3B (UD-Q4_K_M) vs its base Qwen3.5-35B-A3B (Q4_K_M) · llama.cpp, think-OFF, temp 0, native tool-calling · agentic board + SWE-bench Verified anchor (30 bugs, official harness)
**Question:** Qwen-AgentWorld (arXiv 2606.24597) trains a language world model to predict environment state transitions, and claims that this "LWM warm-up" **transfers to agentic tasks with zero agent fine-tuning, +3.4–12.8%**. Does the released LWM-warmed checkpoint actually act as a stronger agent than the base it was built from?

> **Follow-up (`agentworld-think-on-null.md`):** this result is think-OFF, and AgentWorld is built to reason, so a think-ON A/B was the open falsification leg. It's closed: on a bounded 12-bug subset, AgentWorld is bug-for-bug identical whether reasoning is on or off. Thinking mode isn't hiding the transfer either.

## The numbers

| | synthetic agentic board | SWE-bench Verified (real) | empty patches (give-ups) |
|---|---|---|---|
| **Qwen3.5-35B-A3B** (base) | 97.5 | **16/30 (53%)** | 10 |
| **Qwen-AgentWorld-35B-A3B** (LWM-warmed) | 97.5 | **14/30 (47%)** | 13 |

Same checkpoint family, same harness, same think-OFF/temp-0 protocol the base's numbers were measured under. The only variable is the LWM warm-up.

## Finding 1 — the synthetic board can't see it (97.5 = 97.5)

On our native tool-calling Agentic Score, AgentWorld lands **97.5**, identical to the base, with the same 100% task-success and near-identical tool efficiency. This axis saturates near the top (the whole cohort sits in a ~2-point band), so a "flat" here means nothing on its own. It's the reason the rig keeps a real-bug anchor: the synthetic score is where in-distribution gains hide and real differences vanish.

## Finding 2 — on the real anchor, the warm-up is a reshuffle, net −2

Against 30 real SWE-bench Verified bugs graded by the official harness, AgentWorld resolves **14/30 vs the base's 16/30**. It isn't strictly worse, it's a behaviour change: **11 bugs solved by both, 5 base-only, 3 AgentWorld-only** (it newly fixes `django-16429`, `matplotlib-23314`, `sphinx-8621` that the base missed, while losing five others). So the LWM training moved which problems the model can close, but the net is **−2 with more give-ups: 13 empty patches vs 10**, at a high mean of **33 of 40 steps per bug**. It explores longer and commits fewer fixes — the persistence-under-ambiguity tell, not a capability gain.

The claimed +3.4–12.8% transfer lands nowhere on the rig: **0% on the synthetic axis, −12.5% on real resolve.**

## What this is, and isn't

- **It's a controlled A/B, think-OFF.** Both models ran the exact same protocol the base's 16/30 was measured under, so the comparison is clean and the only moving part is the LWM warm-up. Under that protocol, no positive transfer to real agentic coding.
- **It is not a refutation of a think-ON number.** AgentWorld is designed to *reason* about environment transitions and recommends thinking mode (temp 0.6). We held thinking OFF to match the base. A think-ON A/B (both models, re-run) is the real falsification leg and is the queued follow-up; a think-OFF null doesn't speak to a think-ON gain.
- **Scope: the SWE-coding slice.** The paper's +3.4–12.8% spans AgentWorldBench's seven domains (MCP, Search, Terminal, SWE, Web, OS, Android). We tested the one the rig anchors on — real software-bug resolve. No transfer there.
- **Quant:** AgentWorld ran unsloth UD-Q4_K_M vs the base's bartowski Q4_K_M. The dynamic quant is, if anything, slightly higher quality, so the null is conservative against AgentWorld.
- **n=30, single seed:** −2/30 is within noise. The honest headline is "no positive transfer," not a strong regression.

## Worth it if / not if

- **As a drop-in agent, the LWM checkpoint gives a home-lab user nothing over the plain base** on real coding, think-OFF. If you want the best 35B-A3B agent on a 5090 today, the base (or Qwen3.6-27B) is the pick.
- **The interesting result is for environment simulation, not agency.** AgentWorld's actual job is modelling environments (AgentWorldBench); the "use it as an agent for free" transfer is the part that doesn't carry to real bugs here. Its env-simulation quality is a separate, untested question.
- **The think-ON regime is the open door.** If the transfer is real, it should appear when the model is allowed to reason. That's the next run.

## Repro

- Agentic board: `./gate_and_run.sh <gguf> agentworld-35b-a3b` (drains/restores Donald, hard tool-calling gate, then `lib.agentic.native.run_native`). → `results/<slug>/agentic_native.json`.
- SWE anchor: serve at `-c 32768 --jinja`, `swebench-env/bin/python -m lib.agentic.native.run_swebench <slug> swebench_ids_30.txt` → predictions; grade with the official `swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified`. → `results/swebench/<slug>.report.json`.
- Model: `unsloth/Qwen-AgentWorld-35B-A3B-GGUF` (UD-Q4_K_M); base reference `Qwen3.5-35B-A3B` Q4_K_M from `reports/swebench-anchor.md`.
