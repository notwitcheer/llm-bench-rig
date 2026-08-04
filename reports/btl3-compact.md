# BTL-3-Compact on a 5090: a sub-2.5-bit 27B, 13 points under plain Qwen3.6-27B on the same harness

**Rig:** one RTX 5090 32GB (sm_120a) · Bad Theory Labs' forked llama.cpp with the AVQ2 CUDA kernels, built from source in their CUDA 13.0.2 container for arch `89-real;120-real` (Blackwell native FP4) · `badtheorylabs/BTL-3-Compact`, the `BTL-3-Compact-AVQ2.gguf` pack (8.39 GB, sha256 `0a4d9dd…888a`, base `Qwen/Qwen3.6-27B` at commit `6a9e13b`) · greedy (temp 0), thinking off · scored on the same t090 harness as the [quant-tax ladder](quant-tax.md): MMLU/GSM8K/HumanEval, same prompts, same graders, 2026-08-04.

BTL-3-Compact is a sub-2.5-bit 27B pitched as near-lossless and agentic: a claimed 92.2% behavior retention against its full-precision self, 95.12% HumanEval, 88.5% BFCL v4. Its base is `Qwen/Qwen3.6-27B` at the exact commit this rig already built a K-quant ladder on, which makes the comparison unusually clean: the honest quant frontier of the same base is already measured, so the only question is where a different sub-2.5-bit method lands against it. Two reasons to run it: the claims sit on this rig's quant-tax axis, and the vendor had validated exact-GGUF execution only on an RTX PRO 6000, never a 5090. No prebuilt runtime ships for the card, so the AVQ2 fork was compiled from source and the model served on it.

## It runs on the 5090

AVQ2 is a proprietary format; stock llama.cpp cannot decode it, so the fork's CUDA kernels (`btl3-avq.cu`, `btl3-int4.cu`, `btl3-vocab.cu`) were built for sm_120a and the model served on the result. It works: the server reports `CUDA0 : NVIDIA GeForce RTX 5090`, `BLACKWELL_NATIVE_FP4 = 1`, and the model produces correct, coherent output (a valid `tenth_prime` returning 29). This is the **first independent RTX 5090 validation** of the AVQ2 path, and the footprint backs the compression claim: 27B resident in **9.5 GiB**. A 27B model fits 9.5 GiB of VRAM only if its weights are compressed to that footprint, so the sub-2.5-bit claim holds for the file itself. The dry-run manifest confirms the shape: a mixed pack of AVQ2 (5.5 GB), affine-int4 (0.9 GB), a bf16 island (1.2 GB), and vocabulary (0.7 GB), averaging under 2.5 bits per weight.

## The finding

On the same harness, greedy and think-off, BTL-3-Compact scores a **77.3 composite**, which is **13 points below the lowest rung of the K-quant ladder** (Q3_K_M at 90.5) and 15 below Q8. It is off the frontier the same base defines: at 2.46 bits per weight it lands far below where the ladder sits at 3.95 bits, and the gap is not one suite's noise. GSM8K 84.0 vs the ladder's 96.4 at Q3, MMLU 75.4 vs 83.8, HumanEval 72.6 vs 91.5. The measurement is clean: zero empty outputs on every suite, MMLU graded as 240 single letters (no thinking-block truncation), GSM8K on the answer marker for 243 of 250 (no grader-fallback drift).

A second read from the speed column: the compression buys VRAM but not throughput. BTL-3-Compact decodes at **47 tok/s, slower than Q8_0's 53** despite carrying a third of the bytes. On the K-quant ladder decode tracks file size directly (smaller is faster, memory-bound); AVQ2 breaks that, because its mixed-precision unpack is compute-heavy enough to erase the bandwidth advantage a 9.5 GiB model should have.

## The numbers

Same base, same harness, one rig. Decode is single-stream `tg`; VRAM is the served peak; quality is pass@1, greedy, think-off. The ladder rows are the [quant-tax](quant-tax.md) run.

| model | bits/wt | size | decode tok/s | VRAM | MMLU (/240) | GSM8K (/250) | HumanEval (/164) | composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B Q8_0 | 8.50 | 29.0 GB | 52.8 | 27.6 GiB | 86.67 | 97.60 | 92.68 | **92.32** |
| Qwen3.6-27B Q4_K_M | 4.92 | 16.8 GB | 80.4 | 16.8 GiB | 84.58 | 98.00 | 93.29 | **91.96** |
| Qwen3.6-27B Q3_K_M | 3.95 | 13.5 GB | 90.2 | 13.8 GiB | 83.75 | 96.40 | 91.46 | **90.54** |
| **BTL-3-Compact (AVQ2)** | **2.46** | **8.4 GB** | **47.2** | **9.5 GiB** | **75.42** | **84.00** | **72.56** | **77.33** |

Composite is the mean of the three suites. The full five-rung ladder (Q6_K, Q5_K_M) is in the quant-tax report; the three shown here bracket the range.

## Thinking mode: the vendor discourages it

BTL-3's marquee numbers (95.12 HumanEval, 88.5 BFCL) are stated as thinking-mode results, and this run is think-off, so the 72.6 above is not a refutation of the 95.12. It is the same protocol the ladder ran, which is what makes it comparable to the ladder. The thinking-on number is a separate question, and the answer is that it is not reproducible on a standard harness. The model card says so directly: thinking is "currently discouraged because the RL-0013 reasoning policy can become repetitive or fail to terminate on some prompts." A short thinking-on probe reproduced exactly that: on most items the model consumed a 2,048-token MMLU budget or a 3,072-token HumanEval budget entirely inside the reasoning block and never emitted an answer (finish reason `length`, empty content), while a minority terminated normally. Larger budgets are not the fix; 2,048 tokens for a multiple-choice question is already generous. The card publishes no generation settings for the mode. So the headline rests on a reasoning mode the vendor itself flags as unreliable, run with undisclosed sampling, and it does not reproduce under greedy decoding on a neutral harness.

## What's not here (and why)

- **This is not the AVQ2 quantization tax.** BTL-3-Compact is a sub-2.5-bit quant of a BTL agentic finetune, not of plain Qwen3.6-27B, so the 13-point gap to Q3 is a model-plus-method result, not the cost of AVQ2 alone. Their 92.2% retention claim is Compact against full-precision BTL-3, and that A/B needs the full-precision weights (mergeable to Q8, ~29 GB, fits the card) run through the same suites. It was not run here; this report measures the shipped Compact artifact against the base's frontier, not against its own full-precision parent.
- **The agentic axis is untested.** BTL-3's target is tool use (BFCL v4 AST 88.5), which this harness does not measure. These three suites are general capability; a fair test of the agentic claim needs a tool-calling harness (the Hermes agent loop), which is a separate run.
- **Thinking-on is a probe, not a scored run.** The non-termination was reproduced on a handful of items, enough to establish the mode is unstable under greedy on this harness; a full thinking-on grid at the vendor's (unpublished) settings was not attempted.
- **Three suites, not a leaderboard q_avg.** MMLU here is the same 4-subject, 240-row subset the ladder used; the composite is a 3-suite mean and is not comparable to a 5-suite `q_avg` row.
- **Speed is on the vendor's kernels.** 47 tok/s is BTL's AVQ2 CUDA path, not a tuned build; it is a real number for the shipped runtime, not a claim about the format's ceiling.

## Reproduce

```bash
# build the AVQ2 fork (Donald stays up; compile is CPU/nvcc), on capsule with docker
git clone --depth 1 https://github.com/Badtheorylabs/BTL-3 ~/btl3-src
docker build --target build --build-arg CUDA_ARCHITECTURES='89-real;120-real' \
  -f ~/btl3-src/packaging/cuda/Dockerfile -t btl3-cuda-build ~/btl3-src/native/llama.cpp
# extract the staged llama-server + libs (bundle libnccl.so.2 from the base image)

# serve (Donald down first) + run the same t090 harness against it
hf download badtheorylabs/BTL-3-Compact model/BTL-3-Compact-AVQ2.gguf --local-dir ~/btl3/model
LD_LIBRARY_PATH=~/btl3/bin ~/btl3/bin/llama-server -m <AVQ2.gguf> --port 8090 \
  -c 8192 -ngl 999 -ctk f16 -ctv f16 --no-cont-batching
cd ~/benchmark-rig
python3 scripts/quant_tax/speed.py --quant BTL3-Compact --port 8090
for T in gsm8k mmlu humaneval; do python3 scripts/quant_tax/gen.py --task $T --quant BTL3-Compact --port 8090; done
python3 scripts/quant_tax/grade.py results/quant_tax/BTL3-Compact__*.gens.json

# chart (on the Mac, matplotlib on system python3)
python3 scripts/chart_btl3_frontier.py   # -> reports/chart_btl3_frontier.png
```

Model card claims cited from the `badtheorylabs/BTL-3` and `badtheorylabs/BTL-3-Compact` repos (2026-08-04); the RTX PRO 6000-only validation and the thinking-mode caveat are the vendor's own, quoted from those cards.
