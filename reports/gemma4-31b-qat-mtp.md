# The quality king, quantized: Gemma 4 31B QAT + MTP on a 32GB card

**Rig:** RTX 5090 32GB (capsule, llama.cpp b9562) · **Date:** 2026-06-08 · **Task:** local-ai-roadmap t039
**Arms:** `gemma-4-31B-it-qat-q4_0` (QAT) vs naive Q4_0 vs Q6_K (the prior quality king) · MTP drafter `gemma-4-31B-it-MTP-Q8_0`

## The setup

Gemma 4 31B was already this rig's **quality king** — but at Q6_K it **OOM'd at 32K context** on a 32GB card.
Two June-2026 drops promised to fix both ends: Google's **QAT checkpoints** (near-BF16 quality at Q4) and
**MTP** merged into llama.cpp (a draft head for ">2x" decode). So: does QAT hold the crown at a third the VRAM,
does MTP deliver, and does the lighter model finally handle long context? Three things to settle on our own hardware.

## 1. Quality — Q4 barely costs anything, and QAT ≈ naive Q4

q_avg (5 tasks, think-off, 50% MMLU/HellaSwag, full ARC/GSM8K/HumanEval):

| | MMLU | ARC-C | HellaSwag | HumanEval | GSM8K | **q_avg** |
|---|---|---|---|---|---|---|
| **QAT-Q4_0** | 87.25 | 97.61 | 91.60 | 97.56 | 97.27 | **94.26** |
| naive Q4_0 | 87.46 | 97.18 | 90.94 | 97.56 | 97.19 | **94.07** |
| Q6_K (king) | 87.82 | 97.61 | 91.95 | 96.34 | 97.50 | **94.24** |

All three land within **0.19 q_avg of each other.** QAT-Q4 (94.26) *ties* Q6_K (94.24) and beats a dumb Q4_0 by
a noise-level +0.19. The honest read: **Gemma 4 31B is remarkably quantization-robust** — Q4 costs essentially
nothing on standard benchmarks, and QAT's specific edge over naive Q4 doesn't show up here. QAT's value isn't a
better benchmark number; it's that Q4 *at no quality cost* is what unlocks the next two wins.

## 2. Speed — MTP is real: 1.67x decode (2.3x vs the old king)

Decode throughput (tok/s):

- Q6_K: **55** · QAT-Q4: **76** (smaller quant alone) · **QAT-Q4 + MTP: 125**

MTP (draft head, `--spec-type draft-mtp --spec-draft-n-max 4`, 33% draft acceptance) lifts QAT-Q4 from 76 to
**125 tok/s — a 1.67x decode speedup.** Short of the headline ">2x" (acceptance-rate bound at our settings), but
substantial and verified. Stacked against the Q6_K king's 55 tok/s, the QAT-Q4 + MTP combo is **2.3x faster** —
the quant shrink and the draft head compounding.

## 3. Long context — the king finally clears 128K

QAT-Q4 server load at depth, vs Q6_K's prior wall:

| context | QAT-Q4 | Q6_K (prior) |
|---|---|---|
| 32K | ✅ 23.8 GB | ❌ **OOM** |
| 64K | ✅ 26.4 GB | — |
| 128K | ✅ 31.5 GB | — |

**QAT-Q4 loads the full 128K context (31.5 of 32GB) where Q6_K couldn't even reach 32K.** This is the real
payoff of the VRAM cut: not a quality story, a *capability* story. The quality king couldn't do long context on
this card; quantized, it can.

## The wedge (why QAT matters even though it doesn't win on quality)

QAT's benchmark edge over naive Q4 is negligible — but that's not the point. Q4 (either kind) holds Gemma 4 31B's
quality *and* halves the footprint, and that freed VRAM is exactly what buys the **128K context** and the headroom
for the **MTP draft head**. Lighter isn't the feature; lighter is what makes faster-and-longer possible on one card.

## Feasibility notes (Blackwell + the MTP gotcha)

- **llama.cpp rebuilt 9365 → 9562** to land Gemma4 MTP ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)); Donald (the resident agent) verified
  on the new binary before proceeding. The `mistral4`, gpt-oss, gemma archs all still load.
- **The drafter gotcha:** the first community MTP GGUF (`...-assistant`, arch `gemma4_assistant`, underscore) is
  **rejected as "unknown architecture."** The working one is unsloth's `MTP/gemma-4-31B-it-MTP-Q8_0.gguf` (arch
  `gemma4-assistant`, hyphen). Post-merge the MTP-head GGUF format is still settling — match the arch the build expects.
- 31B Q4/Q6 fit 32GB fully (`-ngl 999 -fa on`), no offload.

---
Harness: the rig's `bench.py` quality+speed + `scripts/mtp_speed.sh` (MTP via `llama-server` `/completion` timings)
+ a long-context load sweep. Raw scores: `results/`. think-off, 50% sample — matches the prior Q6_K methodology.
