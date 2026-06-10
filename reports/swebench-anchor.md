# Reality anchor, tightened: the synthetic Agentic Score predicts real SWE-bench *rank* — with one instructive blind spot (Pearson r = 0.50, Spearman ρ = 0.68)

**Rig:** one RTX 5090 32GB · llama.cpp b9562 · native OpenAI tool-calling (`--jinja`) · temp 0
**Update:** this supersedes the first anchor (r = 0.96, n = 12, 4 models). That result was *directional* —
12 smallest-patch instances, only 4 models. Tightening it to **all 7 leaderboard models x 30 harder
instances** revises the headline down and sharpens what the synthetic board actually predicts.

**Setup:** all 7 models drove our native tool-calling loop with **real repo tools** (read / list / search /
edit / run-bash, executing via `docker exec` inside each instance's container) against **30 SWE-bench
Verified bugs** (smallest-patch superset of the original 12, across the same 12 repos) — true agentic
exploration (the model navigates the repo itself; it never sees the hidden tests). Patches graded by the
**official SWE-bench harness** (apply, run FAIL_TO_PASS + PASS_TO_PASS).

## The board

| model | synthetic | real resolve | empty patches | synth rank | real rank |
|---|---|---|---|---|---|
| Qwen3.6-27B | 98.6 | **19/30 (63%)** | 8 | 1 | 1 |
| Qwen3.5-35B-A3B (base) | 97.5 | 16/30 (53%) | 10 | 2 | 2 |
| Qwopus-GLM-18B | 97.1 | 12/30 (40%) | 14 | 3 | 3 |
| Nemotron-Cascade-2-30B | 96.9 | **4/30 (13%)** | **21** | 4 | **7** |
| Kimi-Linear-48B-A3B | 92.9 | 8/30 (27%) | 10 | 5 | 5 |
| Granite-4.1-30b | 92.0 | 6/30 (20%) | 13 | 6 | 6 |
| Nex-N2-mini | 90.4 | 11/30 (37%) | 16 | 7 | **4** |

**Pearson r = 0.50. Spearman ρ = 0.68.** Five of seven models land at the *exact* synthetic-predicted rank.
Pearson is low only because of a single high-leverage outlier: drop Nemotron and r jumps to **0.80**.

## The honest correction

The first anchor's r = 0.96 was real but over-optimistic — 4 models on 12 easy instances is too thin to
trust. The tightened result is the credible one: **the synthetic Agentic Score is a good predictor of real
*ranking* (Spearman 0.68; 5/7 exact), but a moderate predictor of the resolve *rate* (Pearson 0.50)** — and
the gap has a specific, learnable cause.

## What breaks the linear fit (and why it's the interesting part)

**Nemotron-Cascade-2: synthetic #4, real dead-last.** It scores 96.9 synthetically — fluent tool-driving —
then resolves just 4/30 real bugs, with **21 empty patches**. This is *not* a harness artifact: it produced
9 gradeable patches (4 resolved), so `edit_file` demonstrably works for it. The telemetry shows the mechanism
— on every empty instance it ran ~14 tool calls (read/search), then **terminated cleanly without committing
an edit** (`stalled=False`, zero serving errors, avg 15 steps vs 33 to 39 for the models that grind). It
*explores* the repo fluently and then *gives up* — the same fingerprint as Granite, the synthetic board's
known "lean but brittle" model. Synthetic tool-fluency measured the driving; it never measured the willingness
to actually commit a fix on a hard bug.

**Nex-N2-mini: synthetic last, real #4.** The mirror case — underrated synthetically (90.4), quietly resolves
11/30, beating four models ranked above it. Its agentic post-train shows up on real bugs in a way the mock
tasks under-credited.

**The consistent middle held.** Qwen3.6-27B (#1), base Qwen3.5 (#2), Qwopus (#3), Kimi (#5), Granite (#6) all
rank exactly as predicted. Two of the rig's earlier findings survived: **small/merged models punch up**
(Qwopus-GLM-18B, an 18B merge, beats the 48B and a 30B on real bugs) and **lean ≠ robust** (Granite resolves
near the bottom with the fewest steps and the give-up fingerprint).

## Honest caveats

- **n = 30, smallest-patch superset.** Still easier than full Verified, so absolute resolve rates run high;
  the valid claims are the cross-model *correlation* and *ranking*, not the absolute rates.
- **Uniform prompt.** All models got the same loop and the same "fix it with edit_file, then stop" prompt
  (temp 0). A more insistent prompt might coax Nemotron into committing edits — but the comparison is fair
  because the protocol is identical for every model.
- **One outlier, big leverage.** With 7 points, Nemotron alone moves Pearson from 0.80 to 0.50. Reported both
  ways; Spearman (0.68) is the more robust summary.

## Verdict

A synthetic, deterministic agentic bench on one 5090 predicts real SWE-bench **rank** well (Spearman 0.68, 5
of 7 exact) — but it is a *moderate* predictor of resolve rate (Pearson 0.50), and it systematically
**over-ranks models that drive tools fluently but won't commit a fix** (Nemotron is the cautionary case).
Worth using as a cheap rank proxy — *if* you remember it measures tool-driving, not fix-committing, and treat
a fluent-but-low-step model as a give-up risk until a real anchor confirms it.

---

*Supersedes the n=12 / 4-model anchor (r=0.96). Harness: `lib/agentic/native/{tools_repo,run_swebench}.py` in
notwitcheer/llm-bench-rig. Generation = our agent loop in Docker; grading = official `swebench` harness.
Chart: `reports/swebench-anchor.png` (7 points, Pearson r=0.50).*
