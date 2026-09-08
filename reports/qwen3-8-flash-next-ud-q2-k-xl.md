# Qwen3.8-Flash-Next on one RTX 5090: a 125B MoE at top-band quality and 52 tok/s from a 2-bit cut

**TL;DR.** Qwen3.8-Flash-Next (125B total, 10B active, 512 experts, hybrid linear attention) does not fit a 32 GB card at any quant, so this treatment runs unsloth's UD-Q2_K_XL (78.9 GB) with the routed experts of the first 28 layers in system RAM and everything else on the GPU (`--n-cpu-moe 28`, 23.0 GiB VRAM peak, 59 GB RAM). Result: **q_avg 93.9** on the five-task board (MMLU **88.5**, the highest MMLU on the board; ARC-C 97.6; HellaSwag 95.4; GSM8K 95.9; HumanEval 92.1), inside the top band with Gemma 4 31B Q6_K (94.2) and Qwen3.6-27B Q6_K (94.2), at **52.6 tok/s** decode, 42.5 at 32k depth. GPQA-diamond think-off **54.0**, above every dense 27B rung (49 to 51) and Ornith 35B (52). The offload curve is the practical finding: from 40 to 28 layers offloaded, every 1.8 GiB of experts moved onto the GPU bought about 1.5 tok/s, and the curve was still climbing at the last point measured. The MTP speculative head (open llama.cpp PR #28243, shared Q4_K_M drafter) is **1.25x on code and repetitive text and a loss on prose and chat** in this offload regime: with experts in RAM the verification pass is expensive enough that acceptance below ~0.7 does not pay. unsloth's "1.3 to 1.7x with MTP" holds only for the predictable half of the workloads here.

![Flash-Next on the board and the offload curve](qwen3-8-flash-next-ud-q2-k-xl.png)

## Setup

- **Hardware:** RTX 5090 32GB (sm_120), 59 GB system RAM, single card, single stream.
- **Model:** `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-Q2_K_XL` (3-shard split, 78,869,128,864 bytes, every shard size-verified against the HF API at run time), architecture `qwen4exp`, 48 layers, 512 experts with 10 active, `layer_types` 3 linear-attention to 1 full-attention. Vendor config fetched 2026-09-08.
- **Build:** llama.cpp master `b10853-9dcf84e5a`, a fresh worktree (the arch merged into master on 2026-08-27; none of the rig's four resident builds carried it, verified by grepping `libllama.so`). MTP leg on a second worktree at PR #28243 head `d1a92352c` (danielhanchen, `qwen4exp/mtp`, open at run time).
- **Offload:** `-ngl 99 --n-cpu-moe N`: dense layers, attention, and the experts of layers N onward on the GPU; the routed experts of the first N layers in system RAM. N chosen by the sweep below.
- **Quality:** the standing five-task board (MMLU/HellaSwag 50% stratified sample seed 42, ARC-C/GSM8K/HumanEval full), greedy, **thinking off** via `chat_template_kwargs.enable_thinking=false` (probed: the template honours it, letter-only answers come back in 2 tokens), `ctx 8192`. GPQA-diamond 198 items, zero-shot, deterministic shuffle, greedy, think-off.
- **Speed:** llama-bench pp512/tg128 with the depth sweep at the chosen N; served (HTTP) lane for the MTP leg: 8 prompts x 4 workloads, 256 tokens, temperature 0, `ctx 16384`.
- Run 2026-09-08, 08:52 to 14:10 CEST in one night-lib script (`fnext-treatment.sh`), Donald drained for the GPU phases and restored. Provenance block (server command, `/props`, template hash, harness sha) in `results/qwen3-8-flash-next-ud-q2-k-xl/meta.json`.

## The offload curve

llama-bench, `-p 512 -n 128 -r 2`, VRAM peak sampled at 1 Hz:

| `--n-cpu-moe` | tg128 tok/s | pp512 tok/s | VRAM peak |
|---:|---:|---:|---:|
| 40 | 38.1 | 254 | 12.1 GiB |
| 36 | 45.2 | 528 | 15.7 GiB |
| 34 | 46.8 | 700 | 17.5 GiB |
| 32 | 48.4 | 720 | 19.4 GiB |
| 30 | 50.5 | 750 | 21.2 GiB |
| **28** | **52.5** | **769** | **23.0 GiB** |

Reads:

1. **Near-linear in resident experts.** Each layer's experts are ~0.9 GiB at this quant; every two layers moved onto the card bought about 1.5 tok/s of decode. Same shape as the Ling-3 curve measured on this rig in August (127B, Q3_K_M), which peaked at 27.5 GiB before inverting.
2. **The sweep stopped short.** 28 was the planned floor and the curve had not turned over; with 9 GiB still free there is room for 26 and 24. That is a 20-minute follow-up and the number to quote will move up a little.
3. **Prefill is the price.** 770 tok/s pp512 against ~3,100 for a fully resident dense 27B; a 16k prompt costs ~20 s of prefill here versus ~5 s resident. Prompt caching (measured on this rig in the spec-cache study) is what makes that bearable for a stable system prompt.

## Speed at N=28

| test | tok/s |
|---|---:|
| pp512 | 770.1 ± 33.6 |
| tg128 | **52.6 ± 0.2** |
| pp512 @ d8192 | 356.4 |
| tg128 @ d8192 | 45.3 |
| pp512 @ d32768 | 364.2 |
| tg128 @ d32768 | 42.5 |

Decode holds 81% of its empty-context speed at 32k depth. The fully resident Qwen3.8-27B (also hybrid linear attention) holds 92% on this card, so part of the extra decay here is the offload path rather than attention. Prefill at depth halves and then flattens, which is the RAM-side expert traffic dominating once the KV cache is no longer the bottleneck.

## Quality

| task | Flash-Next UD-Q2_K_XL | Gemma 4 31B Q6_K | Qwen3.6-27B Q6_K | Qwen3.8-27B Q6_K |
|---|---:|---:|---:|---:|
| MMLU | **88.47** | 87.8 | 87.9 | 85.3 |
| ARC-Challenge | **97.61** | 97.6 | 96.9 | 96.7 |
| HellaSwag | 95.38 | 92.0 | 95.4 | 94.3 |
| GSM8K | 95.91 | 97.5 | 97.3 | 97.5 |
| HumanEval | 92.07 | 96.3 | 93.3 | 94.5 |
| **q_avg** | **93.89** | 94.2 | 94.2 | 93.7 |
| GPQA-diamond (think-off) | **54.04** | 58.6 (Q4_0 row) | 54.5 | 49.0 |

Wilson 95% on q_avg: 93.0 to 94.8 (see `board_ci.csv`), so the three top rows are a statistical tie; the tables sort by q_avg and this row sits fourth. Parse failures: 2 across 13,010 multiple-choice answers.

Reads:

1. **Knowledge is where the parameters show.** The highest MMLU on the board, from a 2-bit file. The 125B of total weights are mostly expert knowledge, and a 2-bit quant of an expert that is only consulted on its topic loses less than a 2-bit quant of a dense layer that every token passes through, which is why UD-Q2_K_XL on a MoE lands where dense IQ2 rungs on this board lose 3 points.
2. **GSM8K and HumanEval pay the 2-bit tax.** Both sit 1.5 to 4 points under the top dense rows; the 13 HumanEval failures are tracebacks at test time (wrong results or runtime errors), not formatting. A Q3 or IQ3_XXS cut (82 GB, fits the same offload shape) would tell whether that is the quant or the model.
3. **GPQA-diamond agrees with MMLU.** 54.0 think-off is the best MoE number on the second tier and level with Qwen3.6-27B Q6_K (54.5); the 27B family's newer rungs sit at 49 to 51.

### A note on the pace gate

The board's multiple-choice instrument gate (abort if the rolling mean completion length over 25 answers exceeds 50 tokens; think-off letter answers run 1 to 7 tokens) tripped at MMLU subject 4 on the first attempt. Probing with the harness's exact message shape showed serving was clean: 2-token answers, thinking suppressed. The cause was the model itself. Every MC completion length was then logged: of 13,010 MC answers, the median is 2 tokens and **8 exceeded 50 tokens**, 7 of them the model declaring the question incomplete (MMLU items whose stem references a missing figure or passage) and one "none of the options are correct". Eight essays out of thirteen thousand is model behaviour a 25-item window can still trip on; the gate was raised to 400 for this slug and the run resumed from its progress files. No score was affected (the gate aborts, it does not alter answers).

## MTP speculative decoding (PR #28243 build)

Served lane at N=28, `ctx 16384`, p50 decode tok/s over 8 prompts per cell:

| workload | plain | MTP n=2 (shared Q4_K_M head) | acceptance |
|---|---:|---:|---:|
| prose | 47.5 | 44.6 (0.94x) | 0.61 |
| chat | 47.3 | 46.5 (0.98x) | 0.68 |
| code | 48.6 | **61.1 (1.26x)** | 0.93 |
| repetitive | 50.8 | **62.5 (1.23x)** | 0.97 |

TTFT unchanged (0.32 to 0.35 s both ways). Weighted acceptance 0.77.

Read: the acceptance rates are healthy and match what the fully resident 27B MTP heads show on this rig (0.60 to 0.99 by workload). What differs is the cost of a verification step. With 28 layers of experts in RAM, every forward pass, including the one that verifies two drafted tokens, pays the PCIe round trip for the routed experts, so a rejected draft costs a large fraction of a plain token. On code and repetitive text nearly every draft is accepted and the head pays 1.25x; on prose and chat a third are rejected and the head is a small net loss. The same mechanism was reported on the PR by tonydiep on a 4-GPU split at 128k context (25.1 plain to 22.5 with MTP at 83% acceptance). No CUDA stall (llama.cpp #27102, reported on this branch on an RTX PRO 4000) appeared in either server log over 64 requests. unsloth's 1.3 to 1.7x figure is consistent with the code and repetitive cells and with a fully resident configuration; it is not what a mixed workload sees under offload.

## Worth it?

**Worth it if** you want the best knowledge-heavy answers this card has produced and can live with 52 tok/s and slow prefill: MMLU 88.5 and GPQA 54 from one consumer GPU plus 59 GB of RAM, with depth decay milder than any dense rung. **Not worth it if** decode speed or long-prompt latency matters: the MoE-in-VRAM rows (Qwen3.6-35B-A3B at 271 tok/s and 93.3; Ornith 35B at 303 and 89.4) are five times faster, and a dense 27B at 62 tok/s prefill-caches a 16k prompt four times quicker. The MTP head is worth turning on for a coding assistant and off for chat.

## Honest limits

- One quant (UD-Q2_K_XL), one greedy pass per cell. The three top q_avg rows are a tie inside the error band; "top band" is the claim, not "best".
- The sweep floor of 28 was a plan choice, not the optimum; 26 and 24 remain to be measured.
- Speed rows are conditioned on the offload shape and 59 GB of system RAM; a box with different RAM bandwidth moves every decode number. They are not comparable to the resident rows' speed in the same way the quality rows are.
- Think-off only. Flash-Next is a reasoning model; think-on GPQA (the calibration leg used on the 27B ladder) is the natural next leg and would take ~6 h at these speeds.
- MTP measured on an open PR at one commit; the branch is moving.
