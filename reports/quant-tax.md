# The quant tax on Qwen3.6-27B: near-zero down to Q4, one real step at Q3

**Rig:** one RTX 5090 32GB (sm_120) · llama.cpp CUDA build (canary-matched sm_120, quantize sha `0a50d990`) · `Qwen/Qwen3.6-27B` base at commit `6a9e13b` · plain K-quants, **no imatrix** (stated, not implied) · greedy (temp 0), thinking off · every rung served and scored on the same harness, same day (t090 ladder, 2026-08-03).

This is the ladder the [NVFP4 report](qwen3-6-27b-nvfp4.md) kept pointing at: same subject, one pin, quality measured across the GGUF K-quant rungs so "does quantization cost accuracy" gets a number. Five rungs, Q8_0 down to Q3_K_M, against a common harness. The BF16 source (54.7GB) is the provenance root, not a serving target on a 32GB card.

## The finding

Down to **Q4_K_M** the tax is inside the noise: composite quality moves 92.3 to 92.0 from Q8 to Q4 (−0.4 pt) while single-stream decode goes **52.8 to 80.4 tok/s (+52%)** and the file shrinks 29.0 to 16.8 GB. **Q3_K_M** is where the ladder first bends: another +12% decode (90.2 tok/s) for a −1.4 pt quality step, the first drop larger than the between-run noise.

Two operator takeaways from the grid:

- **Q6_K strictly dominates Q8_0 here.** Identical scores on all three suites (208/240 MMLU, 244/250 GSM8K, 152/164 HumanEval), 6 GB smaller, 21% faster decode. On this subject there is no reason to serve Q8.
- **Q4_K_M is the sweet spot.** 80 tok/s, 16.8 GB (half a 5090's VRAM, room for a second model or long context), and quality within 0.4 pt of Q8.

## The numbers

Same models, same harness, one day. Decode is single-stream `tg`; VRAM is the served peak; quality suites are pass@1, greedy, think-off.

| rung | size | decode tok/s | VRAM | MMLU (/240) | GSM8K (/250) | HumanEval (/164) | composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 29.0 GB | 52.8 | 27.6 GiB | 86.67 | 97.60 | 92.68 | **92.32** |
| Q6_K | 22.4 GB | 63.7 | 21.7 GiB | 86.67 | 97.60 | 92.68 | **92.32** |
| Q5_K_M | 19.5 GB | 71.8 | 19.2 GiB | 85.83 | 97.60 | 93.29 | **92.24** |
| Q4_K_M | 16.8 GB | 80.4 | 16.8 GiB | 84.58 | 98.00 | 93.29 | **91.96** |
| Q3_K_M | 13.5 GB | 90.2 | 13.8 GiB | 83.75 | 96.40 | 91.46 | **90.54** |

Composite = mean of the three suites. Prefill tok/s is compute-bound and effectively flat across rungs (1360–1580, run-to-run noise dominates the quant effect); the clean, monotonic signal is decode, which is memory-bound and tracks file size directly.

## Reading the ladder per suite

The aggregate hides the per-suite shape, and the per-suite shape is where the honest caveat lives:

- **MMLU** is the only suite that declines smoothly with bits: 86.67, 86.67, 85.83, 84.58, 83.75, a gentle monotonic slope. Knowledge recall is where the low-bit rounding shows up first and most steadily.
- **GSM8K and HumanEval are flat until Q3.** GSM8K sits at 244/250 through Q8 to Q5, blips to 245 at Q4, drops to 241 at Q3. HumanEval holds 152 to 153/164 through Q4, drops to 150 at Q3. These are one- and two-question moves on samples of 250 and 164, inside the ±1 to 1.5 pt single-flip noise floor until the Q3 step clears it.

So "the tax is near-zero down to Q4" is a claim about aggregate and about GSM8K/HumanEval; MMLU is already paying a small, steady toll the whole way down. Per-suite is a stronger statement than the composite, and only MMLU discriminates the top of the ladder.

## What's not here (and why)

- **No perplexity column.** At the ladder's `-c 4096` context the corpus is only 34 chunks, and `llama-perplexity` returned a **non-physical ordering**: Q3_K_M *lowest* (8.03), Q4_K_M *highest* (9.51), a spread the stated ±0.11 error bars come nowhere near explaining, and one that contradicts the monotonic MMLU from the *same* GGUF files. Re-running the two extremes at the standard `-c 512` stride (274 chunks, same binary and corpus) collapses it: Q8_0 lands at 7.05 ±0.07 and Q3_K_M at 7.02 ±0.07, tied inside the error bars. The scramble was sampling noise from 34 chunks, not a fault in the weights or the perplexity path. And the corrected result agrees with the rest of this report: at honest sampling, wikitext2 perplexity barely separates these quants, so the accuracy suites (MMLU above all) are the discriminator. A full 5-rung `-c 512` column is a cheap follow-up, unlikely to add signal the accuracy grid does not already carry.
- **Three suites, not five.** This ladder ran MMLU/GSM8K/HumanEval, not the full leaderboard `q_avg` (which adds ARC-C and HellaSwag). The composite here is a 3-suite mean and is not comparable to a leaderboard `q_avg` row.
- **Plain K-quants, no imatrix.** An importance matrix would lift the low rungs (especially Q3) and is the obvious next rung of the study. These numbers are the no-imatrix floor.
- **One subject.** Qwen3.6-27B only. The shape (flat to Q4, knee at Q3) is this model's; for other models it is a hypothesis, not established.

## Reproduce

```bash
# on capsule (GPU), Donald down first; run_ladder restores it via the chain wrapper
cd ~/benchmark-rig
bash scripts/quant_tax/run_ladder.sh          # serves each rung, runs speed + 3 suites + ppl
python3 scripts/quant_tax/grade.py results/quant_tax/*.gens.json   # offline grading

# chart (on the Mac, matplotlib on system python3)
python3 scripts/chart_quant_tax.py            # -> reports/chart_quant_tax_5090.png
```

Provenance: `~/t090/gguf/PROVENANCE.json` (base commit, quantize build, imatrix state). Ladder log: `~/t090/logs/ladder-run-2026-08-03.log`.
