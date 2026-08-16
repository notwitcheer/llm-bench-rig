# Qwen3.8-27B: the seven-rung quant-tax ladder

**Date:** 2026-08-14/15 · **GPU:** one RTX 5090 (32GB, sm_120) · **Engine:** llama.cpp b9653 (`9dbc6621a`), CUDA, flash attention on, `-ngl 99` (fully VRAM-resident every rung) · **GGUFs:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (day-0 cuts, UD = unsloth dynamic imatrix recipe) · **Mode:** thinking off, greedy, five-task board harness (MMLU · ARC-C · HellaSwag · GSM8K · HumanEval).

Qwen3.8-27B is the dense-hybrid successor in the Qwen3.x line: `qwen3_5` architecture, 27.32B parameters, linear attention on three of every four layers (per the shipped `config.json`), which is the design change its release notes credit for long-context decode behaviour. Day-0 speed numbers went out on 2026-08-14 ([pp512 3901.5, tg128 77.9 tok/s at Q4_K_M](../dataset/benchmarks.csv)); this report adds the quality board at seven quant rungs, from a 8.4GB 2-bit file to full Q8_0.

## Run context

All seven rungs measured on the same pinned stack (rig harness, llama-server b9653, identical suite prompts, temp 0). Six ladder rungs ran unattended overnight 2026-08-15 (06:54–16:33 CEST, night-lib rails, all six rc=0 with fresh artifacts). The Q4_K_M quality leg is a same-stack rerun banked 18:10 CEST the same day: its original 2026-08-14 day-0 run lost the quality board to a transient DNS failure on the box mid-suite (speed leg unaffected and already published); nothing about the recipe differed in the rerun.

## Results

| Rung | File | VRAM peak¹ | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | q_avg | tg128 |
|------|-----:|-----------:|-----:|------:|----------:|------:|----------:|------:|------:|
| UD-IQ2_XXS | 8.4GB | 10.1 GiB | 82.0 | 95.7 | 92.3 | 95.1 | 89.0 | **90.8** | 114.6 |
| UD-IQ2_M | 9.6GB | 11.3 GiB | 83.7 | 94.9 | 93.7 | 96.7 | 88.4 | **91.5** | 103.3 |
| UD-IQ3_XXS | 11.1GB | 12.8 GiB | 84.3 | 96.3 | 93.8 | 96.8 | 92.1 | **92.7** | 96.2 |
| Q4_K_M | 15.9GB | 17.3 GiB | 85.0 | 96.8 | 94.3 | 97.1 | 92.7 | **93.2** | 78.9 |
| UD-Q4_K_XL | 16.7GB | 18.0 GiB | 85.1 | 96.6 | 94.4 | 97.3 | 93.9 | **93.5** | 76.2 |
| Q6_K | 21.3GB | 22.3 GiB | 85.3 | 96.7 | 94.3 | 97.5 | 94.5 | **93.7** | 63.1 |
| Q8_0 | 27.0GB | 27.6 GiB | 85.2 | 96.8 | 94.4 | 97.4 | 94.5 | **93.7** | 53.1 |

¹ nvidia-smi peak during the llama-bench sweep at default bench context; full per-test prefill rows (pp128–pp16384) in [`dataset/benchmarks.csv`](../dataset/benchmarks.csv).

![ladder chart](chart_q38_ladder.png)

## Reads

1. **No cliff anywhere.** The full spread from Q8_0 to 2-bit XXS is 2.9 q_avg points across a 3.4x file-size range. For contrast, the same harness put Nemotron Lightning's floor rung 1.7 points below its neighbour in one step, and Qwen3.6-27B's ladder paid its first real step at Q3. Here every step down is 0.2–1.2 points, spread across suites rather than concentrated in one.
2. **Q6_K ties Q8_0 to the second decimal** (93.66 both, per-suite deltas ≤0.17). Q8_0 buys 5.7GB of nothing on this board; Q6_K is the effective top of the ladder.
3. **UD-Q4_K_XL is the sweet spot for 24GB cards**: 93.5 at 18.0 GiB peak and 76 tok/s — 0.2 under Q6_K for 4.3 GiB less. Plain Q4_K_M is board-equal (93.2) and the faster pick (79 tok/s) where the 0.3 doesn't matter.
4. **UD-IQ3_XXS is the 16GB-card headline**: 92.7 — one point under Q8_0 — from an 11.1GB file at 12.8 GiB peak and 96 tok/s. A 27B dense model holding 92%+ of the five-task board inside a 16GB card's budget is the strongest small-card result this rig has measured on a dense model.
5. **Even the 2-bit floor holds ~91.** UD-IQ2_XXS keeps GSM8K at 95.1 and loses most of its tax in MMLU (82.0, −3.2 vs Q8). At 115 tok/s and a 10.1 GiB peak it is a usable 27B on a 12GB card at short context, though the IQ2 rungs' HumanEval dip (89.0/88.4 vs 92+ everywhere above) says code is where 2-bit pinches first.
6. **MMLU does most of the falling** (85.2 → 82.0 top to bottom); ARC-C, HellaSwag and GSM8K are near-flat until the very floor. Same shape as the Qwen3.6-27B ladder: knowledge recall pays the quant tax before reasoning does.
7. **Speed is bytes, as always**: tg128 runs 53 → 115 tok/s in near-perfect inverse proportion to file size (memory-bound decode), while prefill stays in one 3.2–3.9k band across the ladder (compute-bound, quant-insensitive).

## Honest limits

- The UD rungs use unsloth's dynamic imatrix recipe; the Q6_K/Q8_0/Q4_K_M rungs are static cuts from the same repo. Recipe and bit-width move together at the bottom of this ladder, so "2-bit costs 2.9" conflates both (the recipe-coverage lesson from the Qwen3.6 NVFP4 rows applies).
- Day-0 GGUFs: unsloth re-cut files in the repo's first hours (the Q4_K_M download died once to a mid-file re-upload). Files here are the cuts as of 2026-08-14 evening; a later re-cut could shift low-rung numbers.
- Thinking-off only, matching the board convention for this model family. VRAM peaks are bench-context peaks, not long-context budgets — the hybrid-attention long-context story (day-0 depth sweep: tg128 −9.6% at d=32768) is a separate axis from this ladder.
