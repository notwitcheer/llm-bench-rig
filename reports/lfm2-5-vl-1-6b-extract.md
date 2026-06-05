# LFM2.5-VL-1.6B-Extract — receipt extraction on one RTX 5090

**Model:** [LiquidAI/LFM2.5-VL-1.6B-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) (1.6B vision-language: 1.2B LM + ~400M SigLIP2 encoder)
**Quant:** F16 model + F16 mmproj (~3.2 GB) · **Engine:** llama.cpp `llama-server` b9365 (`lfm2-vl` arch)
**Hardware:** RTX 5090 32GB · Ryzen 5 9600 · 64GB DDR5 · Ubuntu 26.04
**Dataset:** CORD-v2 `test`, 50 real receipts · greedy decode (`temp 0`) · **first VLM on the rig**

This is the rig's first vision-language benchmark. The text harness (`run_treatment.sh`: MMLU/ARC/HellaSwag/GSM8K/HumanEval) has no image path, so this is a new two-pillar eval — **speed** (extraction throughput) and **quality** (JSON-validity + schema-consistency + field-accuracy vs ground truth).

## Result

| Pillar | Metric | Value |
|---|---|---|
| **Quality** | JSON validity | **100%** (50/50) |
| | Schema-consistency F1 | **1.00** |
| | Field accuracy — total_price | 83.0% (n=47) |
| | Field accuracy — subtotal_price | 83.9% (n=31) |
| | Field accuracy — tax_price | 60.0% (n=20) |
| | **Money-field accuracy (overall)** | **78.6%** |
| | line_item_count exact | 60% |
| **Speed** | Prompt processing (incl. vision) | **~10,500 tok/s** (median) |
| | Generation | **~550 tok/s** (median) |
| | Output length | ~37 tokens (JSON is short) |
| | VRAM | ~3.2 GB (ran alongside a live 27B server) |

## The finding — structure is solved, values aren't

Liquid's two headline metrics reproduce cleanly: **every one of 50 receipts parsed as strict JSON with exactly the requested keys** (their claim: 99.6% JSON validity, 99.6% schema F1). For a 1.6B model that is genuinely impressive — you can wire it into a pipeline and never write a JSON-repair fallback.

But **exact field accuracy against ground truth is ~79%**, and that gap is the story. A guaranteed-parseable wrapper around a value that's wrong one time in five is a different product than the 99.6% headline suggests.

Two real failure modes, read from the raw outputs:
- **Genuine misreads** — `194,000` for a true `174,600`; `27,000` for `23,000`. The vision encoder is small; fine digits on noisy receipts slip.
- **Tax hallucination (why tax is worst at 60%)** — on receipts with no separate tax line, the model fills `tax_price` with the grand total rather than `0`/null. It would rather emit a confident wrong number than break schema. The same instinct that gives 100% validity costs field accuracy.

**Caveat (honest):** field accuracy is a strict exact-digit match against CORD's `gt_parse`; ~3 of the total_price misses are an Indonesian-rupiah decimal/thousands-separator artifact (model emits `63000.0` where gt is `6300000`), not a pure misread. Even counting those, total_price is 83%. `tax_price` n=20 is a small sample.

## Worth it if / not if

- **Worth it** for high-throughput, schema-locked extraction where a downstream check validates the values — triage, pre-fill, "extract then a human/rule confirms." 550 tok/s in 3 GB means you can run a swarm of these next to your main model.
- **Not if** you need unattended money-critical accuracy. 1-in-5 wrong on totals, 2-in-5 on tax, is not a back-office-of-record number.

## Reproduce

```bash
# data (no Pillow needed — raw PNG bytes; llama.cpp decodes)
python scripts/vlm_prep_cord.py ~/vlm-bench/data 50
# serve (own port, leaves your main model up)
llama-server -m LFM2.5-VL-1.6B-Extract-F16.gguf --mmproj mmproj-...-F16.gguf --jinja -ngl 99 -c 4096 --port 8091
# bench
python scripts/vlm_extract_bench.py ~/vlm-bench/data ~/vlm-bench/out --url http://127.0.0.1:8091
```

Raw per-image outputs are kept in `results/lfm2-5-vl-1-6b-extract/vlm_results.json` for offline re-scoring (zero GPU).
