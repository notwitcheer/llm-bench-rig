# Kitty's 2-bit KV cache holds long-context retrieval, the axis its paper never measured

**Rig:** one RTX 5090 32GB (sm_120a) · Kitty (Summer-Summer/Kitty, arXiv 2511.18643), Kitty-Pro config: 2-bit K and 2-bit V, top 20% of Key channels promoted to 4-bit, 32 sink tokens · its reference fake-quant `DynamicCache` on the pinned transformers 4.53.2 fork · `Qwen3-8B` fp16, SDPA, thinking off · Paul Graham essays as needle-in-a-haystack filler · f16 KV vs 2-bit KV A/B, identical prompts, greedy · 2026-08-06.

KV-cache quantization exists to compress long contexts. Kitty reports near-zero loss at 2 bits: on Qwen3-8B its Kitty-Pro variant drops FP16 accuracy by about one point. But read where that number comes from. The seven tasks it is measured on, GSM8K, MATH, HumanEval, GPQA, MMLU, AIME, are short-input, long-generation reasoning tasks. The paper's "long context" means a long chain of thought, up to 32k generated tokens, not a long input to retrieve from. There is no needle-in-a-haystack, no RULER, no passkey anywhere in the paper or the repo.

The axis KV compression is for, holding a long input and recovering a fact buried deep inside it, is the one axis the method was never tested on. This report tests it.

## Calibrate the baseline before trusting any delta

The failure mode to avoid is blaming the quantizer for something the model does on its own. A model that cannot retrieve at 32k with a full-precision cache tells you nothing about the cache when you quantize it. So the f16 baseline runs first, across the whole grid, and only the cells where f16 is clean are worth an A/B.

f16 Qwen3-8B retrieves a single magic-number needle at 100% across every cell of an 8k, 16k, 32k length ladder and five needle depths (0 to 100%). The baseline is clean. One boundary showed up here: 64k does not fit. A one-shot 64k prefill for an 8B, plus its f16 KV cache, peaks past the 32GB card and OOMs. Kitty cannot rescue that case either, because its reference path is a fake-quant simulation that stores the cache dequantized in fp16, so it saves quality information, not memory. 64k on a 32GB card is out of reach for this model regardless of the quantizer. The ladder is capped at 32k, which is already 16x the longest input Kitty's own evaluation ever used.

## Single needle: a ceiling, and why that is weak evidence

With one needle, 2-bit KV matches f16 exactly: 100% versus 100% in all fifteen cells, delta 0.0 everywhere, including the needle placed at the very start of a 32k context where the whole span behind it is quantized. The quantizer is engaged, not silently falling back to fp16: the 2-bit leg runs about 18% slower per query (5.13s versus 4.36s at 32k) because its dequantization kernel runs every step, and a separate check confirmed the stored cache carries 2-bit and 4-bit levels, not 128.

A perfect tie is a weak result. When both legs score 100%, the test cannot tell "the quantizer is lossless" apart from "the task is too easy to stress it." A single obvious needle with an explicit question is an easy retrieval. To trust a null, the task has to be hard enough that something can move.

## Multi-needle: a task with teeth, and the quantizer still holds

The harder probe hides eight distinct magic-number needles at eight spread depths in one context and asks for every one by name. Recall is scored per needle, so partial failures show up, and each needle is bucketed by its depth, so a failure that hits only mid-context or only the deep end is visible. 20 prompts per leg, 160 needle recoveries each, at 16k and 32k.

This task discriminates. The f16 baseline is no longer a ceiling: it slips to 97.5% at 32k and to 90% at the shallowest depth. Something can now move. Through that, 2-bit KV tracks f16 within noise:

| depth % | f16 | 2-bit KV | delta (f16 − kitty) |
|---:|---:|---:|---:|
| 6.25 | 90.0 | 95.0 | −5.0 |
| 18.75 | 100.0 | 100.0 | 0.0 |
| 31.25 | 100.0 | 100.0 | 0.0 |
| 43.75 | 100.0 | 100.0 | 0.0 |
| 56.25 | 100.0 | 100.0 | 0.0 |
| 68.75 | 100.0 | 100.0 | 0.0 |
| 81.25 | 100.0 | 100.0 | 0.0 |
| 93.75 | 100.0 | 100.0 | 0.0 |
| **overall** | **98.8** | **99.4** | **−0.6** |

The only depth where either leg is imperfect is the shallowest, 6.25%, just past the sink region, and it strains both legs, not the quantized one. Across 160 recoveries per leg the 2-bit cache is 0.6 points ahead of fp16, which is about one needle: sampling noise, not a real edge. No depth band degrades under quantization.

## The finding

Kitty's near-zero loss at 2 bits extends to long-context multi-fact retrieval up to 32k, the axis its paper never measured, on a task made hard enough that the fp16 baseline itself starts to slip. No KV-quant degradation is detectable, at any depth. The methodological spine is the point: a single-needle tie proves nothing, so the claim rests on the multi-needle probe, where the task has demonstrated discriminating power and the quantizer still holds.

## What's not here (and why)

- **64k is a hardware wall, not a result.** A 32GB card cannot hold a one-shot 64k prefill for an 8B at fp16 KV, and Kitty's simulation path stores the cache dequantized, so it cannot get under the wall. The retrieval axis beyond 32k on this card is untested, not passed.
- **fp16 is still near the ceiling.** Even the hard probe leaves f16 at 98.8% overall. A harsher task, more needles, similar-value distractors, or retrieval that requires reasoning over the needles, might eventually separate the legs. "No detectable loss" is the honest claim, not "provably zero."
- **The sample is modest.** 160 recoveries per leg. The −0.6-point gap is inside noise; do not read it as the 2-bit cache being better.
- **This is the reference simulator, not a deployed kernel.** The accuracy path fake-quantizes and dequantizes in PyTorch, which is what Kitty ships for quality evaluation. It measures the fidelity of the 2-bit scheme, not the speed or memory of a served deployment.
- **One model, one retrieval task.** Qwen3-8B, magic-number needles, greedy. The result is this model on this axis, not a general law about 2-bit KV.

## Reproduce

```bash
# capsule, Donald drained + restored, ladder capped at 32k (64k OOMs an 8B on 32GB)
cd ~/kitty-bench && source .venv/bin/activate     # torch 2.7.1+cu128, transformers 4.53.2 fork, kitty_sim

# single-needle A/B (f16 first as the calibration, then the 2-bit leg)
python run_niah.py --model Qwen/Qwen3-8B --lengths 8000,16000,32000 --depths 0,25,50,75,100 --needles 5 --leg f16   --out qwen3_8b_f16.jsonl
python run_niah.py --model Qwen/Qwen3-8B --lengths 8000,16000,32000 --depths 0,25,50,75,100 --needles 5 --leg kitty --out qwen3_8b_kitty.jsonl

# multi-needle probe (8 needles/prompt at spread depths, keyed retrieval, per-depth recall)
python run_niah_multi.py --model Qwen/Qwen3-8B --lengths 16000,32000 --M 8 --prompts 10 --leg f16   --out qwen3_8b_multi_f16.jsonl
python run_niah_multi.py --model Qwen/Qwen3-8B --lengths 16000,32000 --M 8 --prompts 10 --leg kitty --out qwen3_8b_multi_kitty.jsonl
python analyze_niah_multi.py --f16 qwen3_8b_multi_f16.jsonl --kitty qwen3_8b_multi_kitty.jsonl --out qwen3_8b_multi_results.json

# chart (Mac, matplotlib on system python3)
python3 scripts/chart_kitty_needle.py   # writes reports/chart_kitty_needle.png
```

Kitty-Pro config (`KittyKVCacheConfig`): `sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=2, promote_ratio=0.2, promote_bit=4, channel_selection=1`. Model `Qwen/Qwen3-8B` fp16, `attn_implementation="sdpa"`, thinking disabled. Filler is the 49 Paul Graham essays (the standard NIAH corpus). The needle harness, the multi-needle probe, and the 64k boundary are the rig's own (2026-08-06); the Kitty-Pro numbers on reasoning tasks are from arXiv 2511.18643.
