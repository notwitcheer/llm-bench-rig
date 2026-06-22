# Qwable-5-27B-Coder: a real-trace coding SFT that ties its base on every cheap eval — and still loses 2 real bugs

**Rig:** one RTX 5090 32GB · llama.cpp 9dbc662 (qwen3) · llama-server `--jinja` native tool-calling · Q6_K, matched vs base (no quant confound) · think-OFF
**Model:** [DJLougen/Qwable-5-27B-Coder](https://huggingface.co/DJLougen/Qwable-5-27B-Coder) — a supervised fine-tune of **Qwen3.6-27B** (dense) on **real agent traces**: "trained first on Claude Fable-5 traces, then continued on Kimi 2.7 Coder traces," aimed at trace-shaped coding behaviour (inspect, decide, edit, verify, recover). Apache-2.0. No public benchmarks; the card reports only "early maintainer runs… on a private coder benchmark."
**Setup:** the same four-leg protocol the rig runs on every Qwen3.6-27B coding tune, all at matched Q6_K vs the model's own base — the 5-task quality suite, the native Agentic Score (40 tool-calling tasks), the SWE-bench Verified reality anchor (30-bug subset, official Docker harness), and MTP drafter speedup by workload. Read against three prior tunes: pi-tune, Qwopus3.6-27B-Coder, Qwable-3.6-27b.

## The numbers (matched Q6_K, think-OFF)

| leg | base Qwen3.6-27B | Qwable-5 | distill effect |
|---|---|---|---|
| quality q_avg | 94.05 | 93.68 | −0.37 (flat) |
| agentic score | 98.61 | 98.61 | 0.00 (identical) |
| **SWE-bench resolved** | **19/30 (63%)** | **17/30 (57%)** | **−2 bugs** |
| **give-ups (empty patches)** | **8** | **10** | **+2** |
| MTP speedup (prose / code / repetitive) | 1.8–2.2× | 1.90× / 2.17× / 2.36× | preserved |

<sub>quality detail: MMLU 87.9, ARC-C 97.1, HellaSwag 95.5, GSM8K 97.0, HumanEval 90.9. agentic detail: task-success 97.2%, tool-efficiency 1.00, stability 100%.</sub>

**Quality flat. Synthetic agentic identical to the base. Real bug-fixing down 2, give-ups up 2.** Two of the rig's three cheap axes don't move at all, and the third moves less than a point — but on the same 30 real bugs the tune resolves fewer and abandons more.

## The cheap axes are blind — again

Three Qwen3.6-27B coding tunes had already shown that a synthetic agentic score can't separate same-base coders (the band is 97.6–100 while real SWE spans 11–20). Qwable-5 is the cleanest case yet: its agentic score is **98.61, to the decimal the base's own number**, with identical task-success (97.2%) and a perfect tool-efficiency. By every fast, re-runnable signal it *is* the base. Only the reality anchor sees the gap: −2 resolved, +2 empty patches. This is the second tune (after Qwable-3.6) where the synthetic score stays flat and the regression hides entirely in persistence-under-ambiguity.

## Real traces are necessary, not sufficient

The interesting part is *why this one should have worked*. The rig's standing finding is that **training-data provenance** predicts real capability: pi-tune — the only tune that ever improved on its base (20/30 vs 19) — was SFT on real non-thinking agent traces, while the synthetic-distill tunes regressed. Qwable-5 is also a real-trace SFT (Fable-5 then Kimi-2.7-Coder agent traces), so the provenance heuristic predicts an improvement. It didn't deliver one. The recipe class that worked for pi-tune (real traces) was not enough on its own here.

| tune | training data | agentic | real SWE | verdict |
|---|---|---|---|---|
| **pi-tune** | real non-thinking agent traces (terminal/repo/DevOps) | 98.01 | **20/30** | improves |
| *base* | — | 98.61 | 19/30 | — |
| **Qwable-5** | real Fable-5 + Kimi-2.7-Coder traces | 98.61 | **17/30** | regresses |
| Qwopus-Coder | Hermes agent traces | 100.0 | 17/30 | regresses (in-distribution) |
| Qwable-3.6 | Fable-5-style distill | 97.64 | 11/30 | regresses (synthetic flat) |

The give-up rate is the same tell as the others: 8 → 10 empty patches. SFT on a particular trace style narrows the policy toward emitting fewer, more cautious trajectories. "Real traces" is not a guarantee — *which* real traces, on *which* task distribution, is the variable. pi-tune's terminal/repo/DevOps trajectories remain the only data shown to move real resolve up.

## The drafter survived — but the capability didn't

Qwable-5 ships an MTP (nextn) draft head, and it **survived the fine-tune**: self-speculative decode runs 1.90× on prose, 2.17× on code, 2.36× on repetitive output (62.7 tok/s base decode → up to 148 tok/s) — squarely in the base's 1.8–2.2× range, and well clear of Qwopus-Coder's degraded 1.4–1.6×. So this SFT kept the drafter that a sibling coder's SFT wrecked.

That makes Qwable-5 a clean dissociation: **a fine-tune can preserve its MTP drafter and still lose real bug-fixing ability.** Drafter-survival and capability-preservation are independent axes. Keeping fast self-speculation tells you the output distribution didn't drift far enough to break the draft head — it tells you nothing about whether the model still solves the bug.

## What it means

- **Don't trust a coding tune's quality or agentic numbers.** Qwable-5 is the strongest demonstration on the board: agentic score identical to the base, quality within a third of a point, and it still resolves fewer real bugs. The fast evals certify "not broken," not "as capable."
- **Provenance is a direction, not a recipe.** Real agent traces are the right ingredient, but the source and task distribution decide the outcome — Fable-5 + Kimi traces regressed where terminal/repo/DevOps traces improved.
- **The give-up rate is the durable signal.** Across every regressing tune the empty-patch count rises; it's the mechanism (cautious, fewer trajectories), not the headline resolve count, that's robust at small-n.

## Honest caveats

- Single seed, 30-bug subset. 17 vs 19 is inside the rig's noise band — the robust reading is "did **not** improve, and give-ups rose," corroborated by the matched agentic-flat, not the exact bug count.
- Benched at Q6_K vs base Q6_K (airtight quant pair). An NVFP4 build is published but unbenched here — a serving-speed follow-up, not a capability question.
- This measures *this* checkpoint's Fable-5 + Kimi-2.7 SFT, not "real-trace SFT" in general — the point is precisely that real-trace provenance did not transfer pi-tune's gain.

## Reproduce

`run_treatment.sh <gguf>` (quality) + `gate_and_run.sh <gguf> <slug>` (native Agentic Score, Donald-safe) + `swe_gen_one.sh`/`swe_grade_one.sh` (SWE-bench gen + official grade, 30-bug `swebench_ids_30.txt`) + MTP: serve the same GGUF with and without `--spec-type draft-mtp --spec-draft-n-max 2` and read `timings.predicted_per_second`. `build_agentic_leaderboard.py` adds the model to the board (META line + re-run). One model, all four legs, ~3h on one 5090.
