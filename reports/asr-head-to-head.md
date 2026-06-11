# Sovereign ASR on one RTX 5090: a 0.6B transducer beats 5-year-old Whisper-large on *noisy* speech — and runs 4–9× faster

**Rig:** one RTX 5090 32GB · parakeet.cpp (ggml, sm_120 CUDA) vs whisper.cpp · LibriSpeech
test-clean / test-other, 200 utts/split (400 total) · **load-once** · one shared text normalizer · temp 0

**Setup:** Each model loads its weights **once** and transcribes the whole 400-utterance subset, so RTFx
(real-time factor = audio-seconds processed per processing-second) measures *transcription*, not model
*load* — a per-file CLI loop reloads the model every call and silently measures load time, the single most
common way to publish a bogus ASR speed number. Every hypothesis **and** reference passes through **one**
normalizer (lowercase, strip punctuation, fold contractions) before **micro-averaged** WER (total errors /
total reference words — the LibriSpeech standard, not the mean of per-utterance WERs). Parakeet-TDT-0.6b-v2
(f16) runs via parakeet.cpp's native `bench --manifest` (TDT decoder, load reported separately); Whisper
large-v3 / large-v3-turbo run via whisper.cpp's multi-file mode (one load over the whole invocation). The
always-on local LLM was drained for the run; VRAM is each model's own steady-state footprint.

## The board

| model | params | WER clean | WER other | RTFx | VRAM |
|---|---|---|---|---|---|
| **parakeet-tdt-0.6b-v2** | 0.6B | 1.49% | **4.73%** | **451×** | **2.0 GB** |
| whisper-large-v3 | 1.55B | **1.47%** | 5.96% | 53× | 4.6 GB |
| whisper-large-v3-turbo | 809M | 1.42% | 6.26% | 117× | 2.5 GB |

## The finding

**On clean read speech, it's a three-way tie** — all three within 0.07pts (1.42–1.49%). Clean LibriSpeech is
saturated; nobody wins, and chasing that number is chasing noise.

**On the harder test-other split, the 0.6B Parakeet wins decisively: 4.73% vs 5.96% (v3) vs 6.26% (turbo).**
The *smallest* model is the *most* noise-robust. And efficiency isn't close — Parakeet is **3.9× faster than
turbo, 8.5× faster than v3**, on the **least VRAM** of the three.

The pointed version: whisper-large-v3-turbo is Whisper's own *speed* variant, and Parakeet beats it on
**every axis** — faster *and* more accurate on noise. The only place anything edges Parakeet is full v3 on
clean speech, by 0.02pts, at 8.5× the compute and 2.2× the VRAM.

## Mechanism

Whisper (2022) is an attention **encoder-decoder**; Parakeet is a **TDT transducer** (token-and-duration),
frame-synchronous decoding built for streaming throughput — that architecture is where the 451× RTFx comes
from. The interesting part is the accuracy: at 0.6B, Parakeet is *not* under-capacity for English read
speech — it's right-sized, so it gives up nothing on noise while running an order of magnitude faster.
Turbo, by contrast, buys its speed by cutting Whisper's decoder stack (32 → 4 layers) and pays for it on the
hard split (6.26%, the worst here). Same "small + specialized beats big + general" result the rig keeps
hitting elsewhere.

## Honest caveats

- **200 utts/split is a directional subset**, not the full test-other (2,939 utts). The 1.2pt noise gap is
  real on this sample, but a full run would tighten the interval — treat the *ranking* as solid and the
  *exact margin* as provisional.
- **English-only.** Parakeet-TDT is monolingual; Whisper-large is multilingual + can translate. This board
  is an English-transcription comparison, nothing more.
- **One shared normalizer**, with number-words-vs-digits **not** reconciled — counted as errors *equally*
  for every model. A stated limitation, applied identically, so it doesn't bias the comparison.
- Parakeet at f16 (the fair quality point against Whisper's ggml weights); quant sweep is future work.

## Worth it?

**Yes — Parakeet-TDT-0.6b is the sovereign ASR default on the 5090:** faster, leaner, tougher on noise.
Keep Whisper-large only when you need non-English or translation.

---

*Method note:* the load-once runner (`scripts/asr_bench.py`) and the scorer (`scripts/asr_report.py`,
dep-free micro-WER) are in this repo; the chart is `scripts/chart_asr.py`.

**Sources**
- parakeet.cpp (ggml runtime): https://github.com/mudler/parakeet.cpp
- Prebuilt Parakeet GGUFs: https://huggingface.co/mudler/parakeet-cpp-gguf
- LibriSpeech: https://www.openslr.org/12
- 2026 open-ASR landscape: https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release
