# "2.5x faster, runs on 24GB": what Unsloth's Qwen3.6-27B NVFP4 actually does on an RTX 5090

**Rig:** one RTX 5090 32GB (sm_120) · vLLM 0.21.0 · torch 2.11.0+cu130 · flashinfer 0.6.8.post1 · `unsloth/Qwen3.6-27B-NVFP4` (compressed-tensors, FP4 weights with FP8 high-precision modules) · greedy (temp 0) · streaming, usage-accurate token counts. **Serving axis only**: no quality numbers in this report, by design (see "What's not here").

The model card claims "2.5x faster" (baseline unstated), "27B on 24GB VRAM", and improved tool calling. We measured the serving half of that on the runtime the card itself targets. The quality half waits for the quant-tax ladder, where it can run against re-banked baselines.

## Three walls before the first token

It does not load out of the box. Each wall is reproducible and each fix is small:

1. **The checkpoint quantises `lm_head` to FP8; vLLM never quantises `lm_head`.** The loader builds a plain head, then dies on the orphan tensor: `ValueError: There is no module or parameter named 'lm_head.weight_scale'`. Upstream this is [vllm-project/vllm#44081](https://github.com/vllm-project/vllm/issues/44081) (reported against v0.22.0, so upgrading does not help) and [unslothai/unsloth#6224](https://github.com/unslothai/unsloth/issues/6224). Fix: dequantise `lm_head` to bf16 offline (exact upcast, `weight.float() * scale`, then cast), drop the scale from the shard and the index. One tensor, +1.27GB on disk.
2. **flashinfer 0.6.8 fails sm_120 arch detection during warmup**: `RuntimeError: No supported CUDA architectures found for major versions [12]`. Fix: `FLASHINFER_CUDA_ARCH_LIST=12.0f` plus a valid `CUDA_HOME`. Same wall we hit in June; unchanged.
3. **The stock engine config OOMs at 32k context on a 32GB card.** Weights take 22.3GiB, and the default activation and graph budgets blow the remaining 9GiB during warmup. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--max-num-batched-tokens 2048`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.92`.

With all three: 32k context fits, with KV room for 111,957 tokens and a 28.9GB VRAM peak.

## The numbers

Prefill sweep (exact-token prompts, tok/s = prompt tokens over TTFT) and decode (128 generated tokens), llama-bench pp/tg semantics:

| point | tok/s | TTFT |
|---|---:|---:|
| pp128* | 768 | 166.6ms |
| pp512 | 8,049 | 63.6ms |
| pp2048 | 10,560 | 193.9ms |
| pp4096 | **10,643** | 384.9ms |
| pp8192 | 10,108 | 810.4ms |
| pp16384 | 9,165 | 1,787.7ms |
| tg128 (decode, single stream) | **54.4** | |

*pp128 carries first-request warmup (its TTFT exceeds pp512's); read the curve from pp512.

Batched decode, separate harness (`vllm bench serve`, random 128-in/256-out, ignore-eos): **52.9 tok/s** at concurrency 1 (TPOT 18.7ms), **401.8 tok/s aggregate at concurrency 8** (TPOT 19.4ms). Eight streams cost 4% per-token latency. The two harnesses agree on single-stream decode (54.4 vs 52.9, different measurement forms).

## Where the FP4 win lives

The kernel path is the real thing: vLLM selects `FlashInferCutlassNvFp4LinearKernel`, and flashinfer JIT-builds native cutlass sm_120 FP4 GEMM kernels. No marlin fallback. That matters because it corrects our own earlier line: the Gemma 4 spec-decode report said consumer Blackwell has no native FP4 kernel. That was true of that model's ModelOpt path under default env. It is not true in general: compressed-tensors NVFP4 plus the arch flag serves real FP4 GEMMs on a 5090.

The speed profile shows the mechanism. Prefill is compute-bound, so the FP4 GEMM pays: 10.6k tok/s at 4k context. Single-stream decode is memory-bound, one token per step, so the compute win never applies: 54 tok/s. Batching restores arithmetic intensity, which is why 8 streams cost 4% latency. This is the same shape we measured in the QuTLASS MXFP4 study: FP4's headline multiplier lives where the matmuls are large.

## "Runs on 24GB": no, not on vLLM

As shipped, the checkpoint does not load on vLLM at all (wall 1). After the mandatory lm_head fix, weights are 22.3GiB. We capped vLLM's budget at 24GiB (`--gpu-memory-utilization 0.765` on the 31.4GiB card, generous to the claim since a real 24GB card pays display and context overhead inside it) and stepped context down: 32768, 16384, 8192, 4096. Every step fails to boot: `ValueError: No available memory for the cache blocks`. Weights plus CUDA context plus activation workspace consume the entire budget before one KV block. Even the unmodified 21.1GiB weights would leave under 2GiB for everything else. The claim may describe the separate MLX build (not tested here). On vLLM it is arithmetically impossible.

## The -Fast variant, explained

Unsloth also ships a `-Fast` variant of the MoE sibling (`Qwen3.6-35B-A3B-NVFP4-Fast`) with no stated difference. The config diff is one block: the base variant keeps `lm_head` **and every expert and shared-expert projection in layers 32 to 39** in high precision; `-Fast` keeps only `lm_head`. The last 8 layers' experts drop to FP4, 26GB becomes 24GB, and whatever late-layer accuracy that protection bought is the price. No weights inspection needed; it is all in `config.json`.

## What's not here (and why)

No MMLU, no HumanEval, no leaderboard row. Our protocol re-banks every baseline under the current pinned harness before any quant comparison runs; we have published one correction already for skipping that step, and we will not ship "2.5x faster" or "same quality" numbers against an unstated baseline. The quant-tax ladder (Q8 to Q2 GGUF, AWQ, and this NVFP4 rung, same subject, same pin) is where those questions get answered.

**Update 2026-07-15:** the re-bank ran. Quality numbers are in the addendum below.

## Addendum (2026-07-15): the quality numbers, with the baseline re-banked

The section above promised no quality numbers until the baseline was re-banked under the pinned harness. That step ran today. Both halves ran under rig commit ec00ff0 with identical seeds (42), identical samples (50% on MMLU and HellaSwag), thinking off, verified behaviorally on both runs: per-item token budgets, pass@1 level, and failure-transcript reads.

The re-bank itself first. The Q6_K baseline (llama-server b9653, the GGUF stack's pin) reproduced its June bank almost exactly: four suites moved by 0.05 points or less, and HumanEval moved 0.6 points, which is one additional passing problem. q_avg went 94.0 to 94.17. The zero point stands, and this time nothing is inherited.

Then the rung: the lm_head-dequanted NVFP4 checkpoint, served on the pinned vLLM stack from the serving section (0.21.0, torch 2.11.0+cu130, flashinfer 0.6.8.post1, native cutlass FP4 path), measured by the same evaluators over the same API.

| suite | Q6_K (llama-server b9653) | NVFP4 (vLLM 0.21.0) | delta |
|---|---:|---:|---:|
| MMLU (50%) | 87.92 | 87.62 | -0.30 |
| ARC-C | 96.93 | 96.42 | -0.51 |
| HellaSwag (50%) | 95.44 | 95.40 | -0.04 |
| HumanEval | 93.29 | 93.29 | 0.00 |
| GSM8K | 97.27 | 97.50 | +0.23 |
| **q_avg** | **94.17** | **94.05** | **-0.12** |

There is no quality cliff in this checkpoint: the whole tax is 0.12 points of q_avg, and the largest single move is 6 answers out of 1172 on ARC-C. The engines differ by design. Each format serves on the stack that runs it natively, both stacks are pinned, and both builds are recorded in the result rows.

One detail the aggregate hides: the two HumanEval scores are identical (153/164) but the failure sets are not. Only 6 of 11 failing problems overlap. The quant fixed five problems the baseline fails and broke five the baseline passes. "Same score" at 4 bits means the errors moved, not that nothing changed; per-task neutrality is a stronger claim than aggregate neutrality, and this checkpoint only earns the aggregate one.

This also sharpens the t070 verdict. On consumer Blackwell, AWQ-int4 beats NVFP4 end-to-end on single-stream speed. Quality is now measured, and it is not the reason: the cost of this rung is the decode profile in the tables above, not the answers.

## Reproduce

```bash
# lm_head dequant (offline, once)
python - <<'EOF'
import json, torch
from safetensors.torch import load_file, save_file
d = "qwen3.6-27b-nvfp4"; shard = "model-00001-of-00005.safetensors"
t = load_file(f"{d}/{shard}")
t["lm_head.weight"] = (t["lm_head.weight"].float() * t["lm_head.weight_scale"].float()).to(torch.bfloat16)
del t["lm_head.weight_scale"]
save_file(t, f"{d}/{shard}", metadata={"format": "pt"})
idx = json.load(open(f"{d}/model.safetensors.index.json"))
del idx["weight_map"]["lm_head.weight_scale"]
json.dump(idx, open(f"{d}/model.safetensors.index.json", "w"))
EOF

# serve
export FLASHINFER_CUDA_ARCH_LIST=12.0f PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve qwen3.6-27b-nvfp4 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 2048 --max-num-seqs 16

# batched decode
vllm bench serve --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 8 --ignore-eos
```

Prefill/decode sweep: `bench.py --speed-only <model dir>` (this repo). Chart: `scripts/chart_nvfp4.py`.
