# Qwopus3.6-27B-Coder, four legs measured: a real #2 quality model, a true 100-tps claim — and the cleanest synthetic-benchmark mirage the reality anchor has caught

**Rig:** one RTX 5090 32GB · llama.cpp b9562 (`--jinja`, temp 0, think-off) · Q5_K_M (the release's claimed quant)
**Model:** [Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF](https://huggingface.co/Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF) — community coder SFT of Qwopus3.6-27B-v2 (itself a Qwen3.6-27B lineage merge), trained on Claude-Opus trace-inversion data **plus Hermes agent reasoning traces**, with a natively fine-tuned MTP head. Benched the day of release (we were download #1). Claims tested: 67% SWE-bench Verified (thinking off), 100 tok/s on a single 5090.

## Leg 1 — quality: the real deal

Standard 5-task treatment (think-off, 50% MMLU/HellaSwag sampling, temp 0):

| MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | q_avg |
|---:|---:|---:|---:|---:|---:|
| 87.5 | 96.8 | 95.2 | 97.5 | 93.3 | **94.1** |

**#2 on the Thinking-OFF leaderboard** — above its own base family (Qwen3.6-27B Q6_K, 94.0)
at a *smaller* quant, a hair under Gemma 4 31B (94.2). The coder SFT gained code (+0.6
HumanEval) and math (+0.2 GSM8K) while giving back only noise-level MMLU (−0.4). No
general-quality tax. Base decode: 70.5 tok/s tg128, 18.2 GiB, 20 GB VRAM peak.

## Leg 2 — MTP: the 100-tps claim holds; the head is weaker than the original

Live-server A/B, draft-depth 2, same protocol as our published Qwen3.6-27B-MTP numbers:

| workload | base | +MTP | speedup | Qwen3.6 original head |
|---|---:|---:|---:|---:|
| prose | 70.3 | 98.0 | 1.4x | 1.8x |
| Q&A | 70.2 | 95.7 | 1.4x | 1.9x |
| code | 70.2 | 97.9 | 1.4x | 2.0x |
| JSON | 70.2 | 112.4 | 1.6x | 2.2x |
| repetitive | 70.2 | 114.5 | 1.6x | 2.2x |

"100 tok/s on a single 5090" — **true** (96–114 typical). But the "natively finetuned"
head accepts *worse* than the original: 1.4–1.6x flat, where the Qwen3.6 head climbs
1.8→2.2x with output predictability. Net effect: **the base Qwen3.6-27B-MTP at Q6_K is
still absolutely faster (113–137 tok/s)** than the coder at Q5_K_M. The SFT presumably
shifted the output distribution away from what the head predicts best — fine-tuning a
model means re-earning its drafter's acceptance. (Caveat: tested at draft-depth 2 for
protocol comparability; a depth sweep might flatter the new head.)

## Leg 3 — Agentic Score: a perfect 100, and why we didn't celebrate

Gate PASS, then **40/40 tasks, tool-efficiency 1.00, stability 100%, 195 tokens/task —
Agentic Score 100.0, new #1** on the [agentic-score-leaderboard](https://huggingface.co/datasets/witcheer/agentic-score-leaderboard),
displacing Qwen3.6-27B (98.6). It is also the leanest agent we've measured (195 tok/task
vs Qwen's 285).

But this model is trained on Hermes agent traces, and our harness speaks that dialect —
partially in-distribution. A perfect score from a model trained on your bench's flavor is
a hypothesis, not a result. Which is what the anchor is for.

## Leg 4 — the reality anchor: the mirage, measured

Same 30 SWE-bench Verified bugs, real repo tools in Docker, official harness grading:

| | synthetic | real resolve | empty patches |
|---|---:|---:|---:|
| **Qwopus3.6-27B-Coder** | **100.0** | **17/30 (57%)** | 8 |
| Qwen3.6-27B (its base family) | 98.6 | **19/30 (63%)** | 8 |
| Qwopus-GLM-18B (sibling) | 97.1 | 12/30 (40%) | 14 |

**The perfect synthetic score loses to its own base on real bugs.** Not the Nemotron
failure mode (it commits patches at the same rate as Qwen — the give-up tell is absent);
this is the second, subtler failure mode: **training-data overlap inflating the synthetic
number without moving real capability.** The 8-model anchor correlation actually
strengthened with this point (Pearson r=0.59, Spearman ρ=0.76 — up from 0.50/0.68), and the
board README now documents both catch modes.

**The 67% claim doesn't reproduce**: our 30-bug subset is *easier* than full Verified
(smallest-patch selection — our baselines run high on it), and the coder scores 57% here.
Wide CI at n=30, but a claim made on the harder full set should not lose ten points on the
easier subset.

## Verdict

A genuinely strong release wrapped in prose that outruns it: real top-2 general quality at
19 GB, a true 100-tps MTP figure, the leanest tool-calling we've measured — and a coding
headline that the official harness doesn't support. **Donald keeps Qwen3.6-27B** (better
real-bug resolve, calibrated board score, no in-distribution asterisk).

Worth it if you want a fast, lean, snappy local agent for interactive coding — the
thinking-off experience their post describes is real. Not if you're choosing models by
their SWE-bench claims — measure, or read someone who did.

## Reproduce

Standard treatment via `run_treatment.sh` · `scripts/bench_mtp_workload.sh` ·
`gate_and_run.sh` (agentic) · `scripts/swebench_batch_{gen,grade}.sh` (anchor) ·
chart: `scripts/chart_qwopus_coder.py`. Board + anchor raw:
[agentic-score-leaderboard](https://huggingface.co/datasets/witcheer/agentic-score-leaderboard).
