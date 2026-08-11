# Qwen3.6-35B-A3B at 2-bit: EschaLabs' Escha-W2, independently measured

**Date:** 2026-08-11 · **Hardware:** RTX 5090 32GB (sm_120) · **Runtime:** EschaLabs escha-sglang wheel 1.0.2+qwen3moe (closed-source SGLang fork) · **Weights:** [EschaLabs/Qwen3.6-35B-A3B-Escha-W2](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) (12.3 GB, `eschamoe` 2-bit experts / int8 dense)

## why this run

Escha-W2 is a 2-bit quantized build of Qwen3.6-35B-A3B with an unusually rigorous vendor card
(paired FP8 protocols, published KLD corpus, a reproduced 4090 grid) — and, as far as I can find,
no independent quality board anywhere. This rig already carries the same base model as a GGUF row
(UD-Q4_K_M, llama.cpp), so a same-model, same-harness cross-check costs one serving window:
their claims, my measurement, one card.

The community pitch that prompted it ("Q8 quality, q3 speed, 1M context") is also checkable.

## method

- Server: their shipped `serve.sh`, exact published 5090 single-user recipe
  (`INT8=on ATTN_BACKEND=triton MEM=0.82 CTXLEN=32768`), `THINK=0`. Server trim: 27.0 GB VRAM.
- Quality: this rig's standard five-task board (MMLU / ARC-C / HellaSwag / GSM8K / HumanEval,
  50% stratified sample on MMLU+HellaSwag, seed 42, temperature 0, thinking off), run over the
  server's OpenAI-compatible `/v1` — same evaluators, prompts, and extraction as every other row.
- Speed: streamed chat completions, decode tok/s = completion_tokens / (total − TTFT), which
  matches the card's own `1000/TPOT` single-stream convention. Three runs per shape, median.
- ~10.5k requests total, zero server failures, one serving window (~50 min).

**Cross-stack caveat, stated up front:** the comparison row (UD-Q4_K_M) is served by llama.cpp and
its 270.6 tok/s is llama-bench tg128; Escha-W2 runs the vendor's own SGLang-fork runtime measured
under chat-server conditions. Quality numbers are apples-to-apples (same harness, same items over
HTTP in both cases); the speed comparison spans two serving stacks — that is the point of the run,
but it is not a kernel-for-kernel quant comparison. Note the conventions differ conservatively:
chat-server conditions typically read *below* llama-bench tg128 (template + sampling overhead), and
Escha-W2 leads anyway.

## results

### quality — same model, same harness, 21.7 GB apart

| build | size | stack | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | q_avg |
|---|---|---|---:|---:|---:|---:|---:|---:|
| UD-Q4_K_M (board row) | 20.6 GB | llama.cpp | 85.0 | 95.7 | 93.3 | 96.7 | 95.7 | **93.3** |
| **Escha-W2 2-bit** | **12.3 GB** | escha-sglang | 84.2 | 95.4 | 92.7 | 96.6 | 93.3 | **92.4** |

The 2-bit build gives up 0.9 q_avg against the Q4 GGUF of the same model. Four of five tasks sit
within 0.8; the loss concentrates in HumanEval (−2.4), exactly the axis the vendor's own card
flags as the one real 2-bit gap (their LiveCodeBench retention: 93.4%). Their honesty holds up.

### speed — vendor claims vs measured

| claim (their card, 5090) | measured here | verdict |
|---|---|---|
| 283 tok/s single-stream decode | 285.7 tok/s (median, ~128-tok prompt) | confirmed, +1% |
| 259.6 tok/s at 2048-token prompt | 262.6 tok/s | confirmed, +1.2% |
| ~25–35 s startup | 25 s to healthy | confirmed |

For board context: the same model as UD-Q4_K_M runs 270.6 tok/s tg128 fully resident. The 2-bit
build is ~6% faster than the Q4 GGUF while serving from a 40% smaller file. TTFT is not quoted:
the single-user recipe leaves the radix prefix cache on, and repeated bench prompts hit it
(first-run 307 ms at a 2k prompt, cached runs 30 ms), so a clean cold-TTFT number needs a
`RADIX=0` pass this window didn't include.

### the community pitch, scored

- **"Q8 quality"** — nearly. −0.9 q_avg vs Q4 of the same model on this board, with the deficit
  almost entirely in code generation. Call it Q4-minus-a-little, not Q8.
- **"q3 speed"** — undersold. It is faster than the Q4 GGUF, not merely Q3-class.
- **"1M context"** — not supported anywhere on the vendor card. Shipped recipes cap at 32k per
  request; RULER is measured to 128k. The "1.4M token pool over 32 agents" reading conflates the
  shared KV pool across concurrent streams with per-request context.
- **"no other 35B for GPU should be run again"** — on a 32GB card, the Q4 GGUF remains the better
  pick: +0.9 quality, same speed class, open runtime. The 2-bit build's real win is the footprint —
  it puts this exact model on 16 GB cards where the Q4 GGUF cannot fit at all.

## honest limits

- The runtime is a closed-source wheel (`escha`, SGLang fork + proprietary CUDA kernels). The
  numbers are real; the kernels are not inspectable. That is a trust decision each operator makes.
- Batched throughput (their ~2,670 tok/s @ bs32 claim) and the 16 GB-card recipes were not
  measured — single serving window, single-stream focus.
- Thinking-ON evals (their MMLU-Pro / MATH-500 / GPQA numbers) were not re-run; this board is
  think-off by construction. Their thinking-on claims remain vendor-reported.
- MC tasks here use generative letter extraction, not loglikelihood — consistent within this
  board, not directly comparable to their commonsense-6 absolute numbers.

## raw

`results/escha-w2-qwen36-35b-a3b/` — `quality_escha.json`, `speed_escha.json`, per-task
`*_detail.json`. Chart: [chart_escha_quadrant.png](chart_escha_quadrant.png).
