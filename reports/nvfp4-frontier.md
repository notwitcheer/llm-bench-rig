# NVFP4 on a 5090: real FP4 lands on the Qwen3.6-27B frontier where K-quants do

**Rig:** one RTX 5090 32GB (sm_120) · llama.cpp CUDA build (t090 pin, build 10108) · `CodeFault/Unsloth-Qwen3.6-27B-NVFP4-GGUF`, the `NVFP4-A` file (20.09 GB, base `Qwen/Qwen3.6-27B`, NVIDIA Blackwell native FP4) · greedy (temp 0), thinking off · scored on the same t090 harness as the [quant-tax ladder](quant-tax.md): MMLU/GSM8K/HumanEval, same prompts, same graders, 2026-08-04.

NVFP4 is Blackwell's native 4-bit float format, and the [quant-tax ladder](quant-tax.md) already measured the honest K-quant frontier of Qwen3.6-27B, so an NVFP4 GGUF of the same base drops straight onto it as one more rung. The question is direct: at its size, does a hardware-FP4 quant match the K-quants, or pay a tax? The June leg (t034) put the NVFP4-vs-Q6 gap at roughly a point of 5-suite q_avg; this re-tests it on the fuller t090 suite, and sits next to the [BTL-3-Compact](btl3-compact.md) result on the same rig, where a different sub-2.5-bit method fell 13 points off this frontier.

## The finding

On the same harness, greedy and think-off, NVFP4-A scores a **92.11 composite**, which lands it **on the frontier between Q5_K_M and Q6_K**. MMLU (86.67, 208/240) and GSM8K (97.60, 244/250) are **identical to Q8_0 and Q6_K**; HumanEval (92.07, 151/164) is one question under Q8's 152. The measurement is clean: zero empty outputs, MMLU graded as 240 single letters, GSM8K on the answer marker for 246 of 250. This is what a lossless-grade 4-bit quant looks like on this suite: the top score on two of three suites, a one-question dip on the third.

## The numbers

Same base, same harness, one rig. Decode is single-stream `tg`; VRAM is the served peak; quality is pass@1, greedy, think-off. The K-quant rows are the [quant-tax](quant-tax.md) run.

| model | size | VRAM served | decode tok/s | MMLU (/240) | GSM8K (/250) | HumanEval (/164) | composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B Q8_0 | 29.0 GB | 27.6 GiB | 52.8 | 86.67 | 97.60 | 92.68 | **92.32** |
| Qwen3.6-27B Q6_K | 22.4 GB | 21.7 GiB | 63.7 | 86.67 | 97.60 | 92.68 | **92.32** |
| Qwen3.6-27B Q5_K_M | 19.5 GB | 19.2 GiB | 71.8 | 85.83 | 97.60 | 93.29 | **92.24** |
| **Qwen3.6-27B NVFP4-A** | **20.09 GB** | **18.1 GiB** | **78.6** | **86.67** | **97.60** | **92.07** | **92.11** |
| Qwen3.6-27B Q4_K_M | 16.8 GB | 16.8 GiB | 80.4 | 84.58 | 98.00 | 93.29 | **91.96** |
| Qwen3.6-27B Q3_K_M | 13.5 GB | 13.8 GiB | 90.2 | 83.75 | 96.40 | 91.46 | **90.54** |

Composite is the mean of the three suites.

## Two mechanisms worth keeping

**NVFP4's decode sits on the memory-bound line.** It runs at 78.6 tok/s from 18.1 GiB of served weights and KV, which is exactly where the K-quant curve predicts for that footprint (Q5_K_M at 19.2 GiB does 71.8, Q4_K_M at 16.8 GiB does 80.4). Decode on this rig is memory-bound, so tok/s tracks resident size, and NVFP4 obeys that: its dequant is cheap enough on Blackwell's FP4 units to add nothing measurable. This is the direct contrast with [BTL-3-Compact](btl3-compact.md), whose AVQ2 unpack is compute-bound and broke the same rule, decoding slower than Q8 despite a third the bytes. The dequant cost is the whole difference: NVFP4's is cheap on Blackwell's FP4 units and adds no measurable decode penalty, while AVQ2's mixed-precision unpack is heavy enough to cost throughput.

**It confirms NVFP4 is near-lossless, and refines the delta.** On the t090 three-suite composite the NVFP4-vs-Q6_K gap is 0.21 points, inside the run-to-run noise the [quant-tax ladder](quant-tax.md) measured. June's t034 put it near a point on the five-suite q_avg; the fuller suite and this specific build differ, but both say the same thing, that hardware FP4 is a near-lossless 4-bit point on Qwen3.6-27B rather than a quality cliff. Whether NVFP4 wins or loses on throughput against AWQ int4 under vLLM is a separate stack and a separate question, untouched here.

## The BTL-3 contrast

Same 5090, same base, same harness, two low-bit methods. NVFP4 lands on the frontier, its quality indistinguishable from Q6 and its speed on the memory-bound line. BTL-3-Compact's sub-2.5-bit AVQ2 fell 13 points below the frontier's lowest rung. At 20 GB NVFP4 gives the quality its size implies; at 8 GB BTL-3-Compact does not.

## What's not here (and why)

- **The 20.09 GB file carries an inert draft head.** The NVFP4-A build preserves a BF16 MTP (multi-token-prediction) draft head for speculative decoding, so the on-disk file is larger than the served model. Single-stream with no draft flags, that head is not loaded, and the served footprint is 18.1 GiB, which is what the table and chart use. A spec-decode run that activates the head is a separate measurement.
- **One NVFP4 variant.** The companion `NVFP4-Q8` file (23.2 GB, more tensors at Q8) was not run; this is the NVFP4-A point only.
- **Three suites, not a leaderboard q_avg.** MMLU here is the same 4-subject, 240-row subset the ladder used; the composite is a 3-suite mean and is not comparable to a 5-suite `q_avg` row (which is why the June delta and this one are not the same number).
- **One subject.** Qwen3.6-27B only; the on-frontier result is this model's, not a general law about NVFP4.

## Reproduce

```bash
# on capsule (GPU), Donald down first; restore on exit
hf download CodeFault/Unsloth-Qwen3.6-27B-NVFP4-GGUF Unsloth-Qwen3.6-27B-NVFP4-A.gguf --local-dir ~/t090nvfp4
~/t090/llama.cpp/build/bin/llama-server -m ~/t090nvfp4/Unsloth-Qwen3.6-27B-NVFP4-A.gguf --port 8090 \
  -c 8192 -ngl 999 -ctk f16 -ctv f16 --no-cont-batching
cd ~/benchmark-rig
python3 scripts/quant_tax/speed.py --quant NVFP4-A --port 8090
for T in gsm8k mmlu humaneval; do python3 scripts/quant_tax/gen.py --task $T --quant NVFP4-A --port 8090; done
python3 scripts/quant_tax/grade.py results/quant_tax/NVFP4-A__*.gens.json

# charts (on the Mac, matplotlib on system python3)
python3 scripts/chart_nvfp4_headtohead.py  # -> reports/chart_nvfp4_headtohead.png (vs Q6_K)
python3 scripts/chart_nvfp4_frontier.py    # -> reports/chart_nvfp4_frontier.png (on the frontier, BTL-3 contrast)
```

Ladder provenance: `~/t090/gguf/PROVENANCE.json`. NVFP4 file: `CodeFault/Unsloth-Qwen3.6-27B-NVFP4-GGUF`, base Qwen3.6-27B.
