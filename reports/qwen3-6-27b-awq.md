# AWQ-int4 on Qwen3.6-27B: the code tax follows recipe coverage, not format — and one engine can't serve it at all

**Rig:** one RTX 5090 32GB (sm_120) · vLLM 0.21.0 · torch 2.11.0+cu130 · `QuantTrio/Qwen3.6-27B-AWQ` (true AWQ: 4-bit, group 128, zero-point, gemm; 8 modules ignored — a **flat** recipe; 21.9GB, 8 shards) · awq_marlin kernel · five-suite quality protocol (MMLU / ARC-C / HellaSwag / HumanEval / GSM8K, 50% sampling on MMLU+HellaSwag, think-OFF via `chat_template_kwargs`, temp 0) against the 2026-07-15 re-banked Q6_K baseline. **Quality axis only** — the serving/speed leg is blocked, and that story is half this report.

Two days ago we showed two artifacts shipping as "NVFP4" for this model sit 0.85 q_avg apart, and that the gap is recipe coverage — which modules stay high-precision — not the format. That was one format, two recipes. Today's rung is the cross-format test: a *different* 4-bit format with the same *flat* recipe shape. If the recipe rule is real, flat AWQ should land near flat NVFP4, far from protected NVFP4. It does.

## The numbers

| task | Q6_K baseline (re-banked) | NVFP4 protected (303 modules HP) | **AWQ-int4 flat (this run)** | NVFP4-GGUF flat (June) |
|---|---:|---:|---:|---:|
| MMLU | 87.92 | 87.62 | **87.50** | 87.0 |
| ARC-C | 96.93 | 96.42 | **96.42** | 96.7 |
| HellaSwag | 95.44 | 95.40 | **95.18** | 94.9 |
| GSM8K | 97.27 | 97.50 | **97.27** | 97.1 |
| HumanEval | 93.29 | 93.29 | **89.02** | 90.2 |
| **q_avg** | **94.17** | **94.05** | **93.08** | 93.2 |

Three reads, each strengthening a thread the ladder already carries:

1. **The tax is code, again.** Every non-code suite sits within 0.0–0.5 points of baseline — inside the noise band of this protocol. HumanEval drops 4.3 points. Same shape as the flat NVFP4-GGUF finding.
2. **Recipe coverage beats format name, now confirmed cross-format.** The two flat recipes land at 93.08 and 93.2 — different formats (AWQ-int4 vs NVFP4), different engines (vLLM vs llama.cpp), different sizes (21.9GB vs 14.6GB), same cluster. The protected recipe holds the baseline exactly at 94.05. At matched coverage, the format contributed approximately nothing. Note the size column: the flat AWQ is nearly the protected build's size (21.9 vs 23.4GB) and still pays the full flat-recipe tax. It is not the gigabytes — it is which modules they're spent on.
3. **Even a real regression is a reshuffle.** AWQ fails 18 HumanEval problems to the baseline's 11: 8 shared, 10 new breaks, 3 problems the baseline fails that AWQ *fixes*. The NVFP4 addendum found the same churn under an identical aggregate (153/164 both, 6/11 shared); this rung shows the churn persists when the aggregate genuinely moves. A 4-point delta understates how much the failure surface shifted.

## The engine that couldn't: sglang 0.5.14 vs this artifact

The quant-tax kickoff pinned the AWQ rung to sglang. That leg is dead, and how it died is worth more than a footnote:

- **At bf16** (`--dtype bfloat16`, required because the checkpoint ships no `torch_dtype` and fp16 crashes — below): the engine loads, reports the awq_marlin kernel, serves single-turn prompts correctly — and produces **degenerate output on anything longer**. Few-shot MC questions come back as rambling delirium that still happens to contain letters. The quality suite banked MMLU subjects at 2–4% with 44–62 unparsed answers each before an unrelated network error killed the run.
- **At fp16** (`--dtype float16 --mamba-ssm-dtype float16`): hard crash at first prefill — `Index put requires the source and destination dtypes match, got BFloat16 for the destination and Half for the source` in `gdn_backend.py`. Qwen3.6 is a hybrid linear-attention (GDN) architecture, and sglang 0.5.14's GDN conv-state buffer is bf16 regardless of model dtype; the mamba dtype flag doesn't govern it.
- **Ruled out:** the chat template (offline `apply_chat_template` renders byte-identical to the known-good NVFP4 checkpoint's) and the request path (the correctly-rendered string fed to `/v1/completions` degenerates the same way).
- **Same artifact, same prompts, vLLM 0.21:** clean, correct, 1–2-token answers.

So the banked numbers above come from vLLM — which is also the engine the NVFP4 rung ran on, keeping the ladder single-variable. The sglang wall goes in the availability column, where it is a finding in its own right: *a quant's quality is a property of the artifact-engine pair, not the artifact.* We keep re-learning this in new costumes — last time as a version-bound loader wall (vllm#44081), this time as a dtype wall that doesn't crash.

## The instrument lesson

The near-miss deserves its own paragraph. sglang's degenerate output *parsed* — the answer extractor found letters in the delirium, graded them, and produced plausible-looking single-digit accuracies. If the run hadn't died on a transient HF Hub 504 three subjects in, the suite would have completed and banked "AWQ scores 2% on MMLU" — a number wrong by ~85 points, from a serving bug, wearing the costume of a catastrophic quant failure. The tell that catches this class automatically is *pace*: think-OFF multiple-choice answers are 1–7 tokens; the poisoned run averaged ~850 tokens per answer. A completion-length gate on MC tasks is now on the rig's TODO. (This is CP43's lesson generalized: separate the instrument's failure mode from the subject's before the number means anything.)

## What's not here

- **Speed/serving axis:** blocked on sglang's GDN path for this artifact; awq_marlin single-stream and batch numbers ride the next rung pass once the leg unblocks (or the sglang pin is revisited).
- **The real-capability axes:** the agentic board and the SWE-bench-30 anchor run as Phase A nights per the quant-tax spec — the q_avg suite is the cheap continuous signal, not the verdict.
- **Recipe-protected AWQ:** cyankiwi ships compressed-tensors AWQ builds of this model keeping 208 and 400 modules high-precision. The recipe rule predicts they recover the code tax the way protected NVFP4 does. That's now a testable prediction sitting one rung away.

Chart: [recipe coverage vs HumanEval](chart-awq-recipe-coverage.png). Ladder rows and protocol: the [leaderboard](../dataset/README.md). Recipe-coverage part one: [NVFP4 addendum](qwen3-6-27b-nvfp4.md).
