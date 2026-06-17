# Qwen3.6-27B pi-tune: the coding tune that actually works — real agent traces beat synthetic distill

**Rig:** one RTX 5090 32GB · llama.cpp 9dbc662 (qwen35) · llama-server `--jinja` native tool-calling · Q6_K, matched vs base (no quant confound)
**Model:** [bytkim/Qwen3.6-27B-MTP-pi-tune](https://huggingface.co/bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF) — a full QLoRA SFT of **Qwen3.6-27B** on **real non-thinking agent traces** (terminal/shell, tool-calling, code editing, repo work, DevOps), with an integrated MTP draft head. Apache-2.0. Honest card: Terminal-Bench 2.0 pass@1 = 28.09, other coding evals "not finalized."
**Setup:** four legs at matched Q6_K vs its own base — quality (5-task), native Agentic Score (40 tool-calling tasks), the SWE-bench Verified reality anchor (30-bug subset, official Docker harness), and MTP speedup by workload. Read against two prior Qwen3.6-27B coding tunes: Qwopus3.6-27B-Coder and Qwable-3.6-27b.

## The numbers (matched Q6_K)

| leg | base Qwen3.6-27B | pi-tune | vs base |
|---|---|---|---|
| quality q_avg | 94.05 | 93.30 | −0.75 (flat) |
| agentic score | 98.61 | 98.01 | −0.60 (flat) |
| **SWE-bench resolved** | 19/30 | **20/30** | **+1, give-ups 8 → 6** |
| MTP speedup (per workload) | 1.8–2.2× | **2.0–2.4×** | preserved/improved |

**Every cheap eval is flat. Real bug-fixing went up.** pi-tune resolves one more of the same 30 bugs than its base — it solves **2 the base couldn't** (`astropy-14539`, `sympy-23950`) and drops 1 (`requests-5414`) — with a lower give-up (empty-patch) rate. And its MTP drafter **survived the fine-tune** (2.0–2.4×, climbing with predictability) where a sibling coder's collapsed to a flat 1.4–1.6×.

## The cross-model finding: provenance, not the label

Three Qwen3.6-27B coding tunes now, same base, same anchor:

| tune (training data) | agentic | real SWE | verdict |
|---|---|---|---|
| **pi-tune** (real non-thinking agent traces) | 98.01 | **20/30** | improves |
| Qwopus-Coder (Hermes agent traces) | 100.0 | 17/30 | regresses + in-distribution mirage |
| Qwable-3.6-27b (Claude Fable-5 distill) | 97.64 | 11/30 | regresses, synthetic missed it |
| *base* | 98.61 | 19/30 | — |

**The synthetic agentic score is a narrow 2.4pt band (97.6–100) across all three — yet real SWE spans 11 → 20.** The cheap axis cannot separate them; pi-tune even has the *lowest* q_avg of the group while resolving the *most* real bugs. The variable that tracks real capability is **what the model was trained on**: real agent traces improved it; synthetic Claude/Hermes distill regressed it. The "agentic coder" label is not the signal — the data provenance is.

## What it means

- **Real traces preserve capability; synthetic distill narrows it.** The give-up rate is the tell — pi-tune gives up *less* (6 vs base 8), the distill tunes *more* (Qwable 13). SFT on real terminal/repo trajectories taught it to keep solving; SFT on synthetic agent dialogue taught a cautious style that quits.
- **The drafter is a second capability the SFT can wreck — or keep.** Fine-tuning re-rolls the MTP head's acceptance (the drafter must re-earn the new output distribution). Qwopus-Coder's dropped to 1.4–1.6×; pi-tune's held at 2.0–2.4×. Same split as quality and SWE: the real-trace tune kept what the synthetic tunes lost.
- **A modest, honest win.** +1 bug is within single-seed noise — but the *direction* (resolve up, give-ups down, drafter intact) is consistent across legs, and it's the only one of three tunes pointing that way.

## Honest caveats

- Single seed, 30-bug subset — the direction is the signal, not the exact count. The +1 is a real +2/−1 swap, not a tie.
- MTP "speedup" is the rig's proxy (vs a fresh non-MTP base baseline at 62 tok/s); I did not read the raw acceptance counter, so I report "preserved/strong," not the card's exact 78%.
- The model is think-OFF-tuned; the agentic codeact/multistep axes use `think:true` (same config as base for fairness — may slightly understate it).
- This is *this* checkpoint's real-trace SFT vs *those* synthetic-distill recipes — a provenance contrast across three tunes, not one controlled variable.

## Reproduce

`gate_and_run.sh <gguf> <slug>` (Agentic Score) + `swe_gen_one.sh`/`swe_grade_one.sh` (SWE-bench, 30-bug `swebench_ids_30.txt`) + `run_treatment.sh` (quality) + `bench_mtp_workload.sh` (MTP speedup) + `build_agentic_leaderboard.py`. One model at matched Q6 vs the cached base, ~5h on one 5090.
