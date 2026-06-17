# Qwable-3.6-27b: the distill every cheap eval passes and SWE-bench fails

**Rig:** one RTX 5090 32GB · llama.cpp 9dbc662 (qwen35) · llama-server `--jinja` native tool-calling · Q4_K_M, base vs distill matched (no quant confound)
**Model:** [Mia-AiLab/Qwable-3.6-27b](https://huggingface.co/Mia-AiLab/Qwable-3.6-27b) — a full finetune of **Qwen3.6-27B** (dense) on a *"cleaned Fable-5-style reasoning/instruction"* dataset, aimed at "deliberate, structured, trace-like" coding behavior. Experimental, no published evals. Ships **only as a Q4_K_M GGUF** (no safetensors).
**Setup:** the rig's native Agentic Score (40 tool-calling tasks) + the SWE-bench Verified reality anchor (30-bug subset, official Docker harness) + the 5-task quality suite. Controlled vs its own base **Qwen3.6-27B at the same Q4_K_M** — and because Qwable only exists at Q4, we benched the base at Q4 too, so any gap is the distill, not the quant.

## The numbers (matched Q4_K_M)

| axis | Qwen3.6-27B base | Qwable-3.6-27b | distill effect |
|---|---|---|---|
| quality q_avg | 94.05\* | 93.40 | −0.65 (flat) |
| agentic score | 98.19 | 97.64 | −0.55 (flat) |
| **SWE-bench resolved** | **18/30 (60%)** | **11/30 (37%)** | **−7 bugs** |
| **give-ups (empty patches)** | **7** | **13** | **+6** |

<sub>\*base quality is Q6_K (we skipped base-Q4 quality); quant moves q_avg <1pt. Agentic task-success is **identical** at 97.2% — the score gap is tool-efficiency only.</sub>

**Static quality flat. Synthetic agentic flat. Real bug-fixing fell off a cliff.** The distill resolves 7 fewer of the *same* 30 bugs and gives up almost twice as often — while two cheaper benchmarks shrug.

## Quant is not the excuse

Qwable only ships at Q4_K_M, so we benched the base at Q4 too. Q6→Q4 on the base cost **1 bug** (19→18) and merely *reshuffled* ~3 instances — a minor, near-random perturbation. On agentic the quant cost (−0.42, 98.61→98.19) and the distill cost (−0.55) are the same negligible order — *neither* cheap benchmark separates these models. Only the real bugs do: the −7 is the post-training, full stop.

## What it means

- **The give-up tell, hidden.** The 8 bugs the base solved that Qwable abandoned span **6 projects** (astropy, matplotlib ×2, requests, xarray ×2, scikit-learn, sphinx) — distributed, not a fluke. The empty-patch rate jumping 7→13 is the mechanism: the Fable-5 trace-style SFT taught it to emit fewer, more cautious trajectories, not to solve more. Same failure family as last session's MoE Qwable-v1 (also landed 11/30).
- **But this time the synthetic score lied.** The MoE Qwable-v1's agentic score *declined* (96.25) and honestly predicted its real drop. This dense one's agentic score stays **flat** (97.64 vs 98.19, identical 97.2% task-success) — it sails through the in-distribution tool-calling tasks and only face-plants on the persistence-under-ambiguity that real bugs demand. So the agentic board is now documented to *sometimes* miss the regression. Third entry in the catalog: nemotron drives-but-never-commits, qwopus in-distribution-mirage, and now **distill-flat-synthetic, real-give-up**.
- **The base is the story's other half.** Vanilla Qwen3.6-27B at Q4 still resolves 18/30 — top-tier, matching the dense base's reputation. The "agentic coder" relabel made a strong base worse on the one axis that's hard to fake.

## Honest caveats

- Single seed, 30-bug subset — the *direction* (−7 resolved, give-ups up) is the robust signal, not the exact counts. base-Q4 vs Qwable-Q4 is the airtight pair; base quality is Q6 (quant <1pt).
- Different recipe/size from the MoE Qwable-v1 (35B-A3B, Q5, explicit Opus-4.7 distill stage). The cross-model "synthetic honest vs synthetic fooled" contrast is an observation across two distills, not a controlled pair.
- This measures *this* checkpoint's Fable-5-style SFT, not "agentic distillation" in general.

## Reproduce

`gate_and_run.sh <gguf> <slug>` (native Agentic Score, Donald-safe) + `swe_gen_one.sh`/`swe_grade_one.sh` (SWE-bench gen + official grade, 30-bug `swebench_ids_30.txt`) + `run_treatment.sh` (quality) + `build_agentic_leaderboard.py` (META line + re-run). Two models (base + distill) at matched Q4, ~5h on one 5090.
