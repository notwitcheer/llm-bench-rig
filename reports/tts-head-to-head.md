# Sovereign TTS head-to-head: the 1.7B Apache model beats the 4B research-license model on a 5090

**Rig:** one RTX 5090 32GB (sm_120) · both models bf16, **neither compiled** · Seed-TTS-eval EN (150 utterances) · voice-clone from each utterance's reference clip
**Models:** [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (Apache-2.0, the `qwen-tts` package) vs [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) (4B dual-AR, Fish Audio **Research** License, from-source `fish-speech`)
**Metrics (4 axes):** round-trip WER (synth → `whisper-large-v3` → WER vs target), SIM-o (WavLM-large+ECAPA cosine, the Seed-TTS/F5-TTS metric), RTFx (audio seconds ÷ synth seconds), first-audio latency. Same harness, same subset, same judge for both.

## The numbers (both n=150)

| axis | Qwen3-TTS-1.7B | Fish-S2-Pro (4B) | winner |
|---|---|---|---|
| round-trip WER | 0.6% | 0.6% | ≈ tie |
| SIM-o (speaker clone) | **0.699** | 0.625 | Qwen |
| RTFx (× realtime) | **2.22×** | 0.39× | Qwen (**5.7×**) |
| first-audio latency | **1.72s** | 9.74s | Qwen |

**Both are equally intelligible. The small one is ~6× faster, clones a touch better, and answers ~6× sooner.** On a consumer 5090, out-of-the-box, the 1.7B Apache model is the better sovereign pick.

## The mechanism, not just the scores

- **WER is a tie because both models are simply good.** Round-trip WER lands at 0.6% for each — Fish's "sub-1% WER" claim *holds*, and Qwen matches it. Intelligibility is not where these two separate.
- **Speed is where the architectures diverge — and where the deployment story lives.** Fish-S2-Pro is a 4B dual-AR model whose serving stack is built for SGLang + `torch.compile` + datacenter cards (the vendor RTF<0.5 number is H100/H200-class). Run it the way you'd run any model out-of-the-box on a 5090 — bf16, no compile — and it generates at **0.39× realtime** (≈3× slower than the audio it's making). Qwen3-TTS-1.7B, same conditions, runs at **2.22×**. That's a 5.7× gap from model size + serving assumptions, not from quality.
- **Latency compounds it.** First audio at 1.72s (Qwen) vs 9.74s (Fish) — the 4B's longer generate makes it unusable for anything interactive without the compiled path.
- **SIM-o: a real but modest edge to Qwen** (0.699 vs 0.625). Both clone faithfully from a single reference; Qwen's embeddings sit closer to the prompt speaker.

## Caveats (read these before quoting the numbers)

- **Neither model was compiled.** This is an honest out-of-the-box, single-5090 comparison. Fish's `api_server` supports `--compile`; with it, Fish's speed would improve materially (toward its H200 claim). The apples-to-apples choice here was no-compile for both. A compiled-Fish follow-up is the obvious next measurement.
- **Round-trip WER uses whisper-large-v3 as the judge** — comparable *between* these two models on the same subset, not directly comparable to either vendor's own WER protocol.
- **SIM-o is the standard WavLM-large+ECAPA** (`wavlm_large_finetune.pth`), the same metric the Seed-TTS / F5-TTS / CosyVoice papers report.
- Licenses differ: Qwen3-TTS is **Apache-2.0**; Fish-S2-Pro is **research/non-commercial**. For a sovereign, ship-it stack that matters as much as the numbers.

## Worth it if / not if

- **Reach for Qwen3-TTS-1.7B** if you want a fast, permissively-licensed voice-clone TTS that runs on one consumer GPU today. It is the out-of-the-box winner here on every axis that isn't a tie.
- **Reach for Fish-S2-Pro** if you can pay the `--compile`/datacenter serving path and want its fine-grained inline prosody control (`[whisper]`, `[excited]`, free-form tags) — features this intelligibility/speed bench doesn't measure. Out-of-the-box on a 5090, it is not the pick.

## Repro

- Synthesis adapters: `scripts/tts_synth_qwen3.py` (qwen-tts, SDPA on sm_120), `scripts/tts_synth_fish.py` (fish-speech ormsgpack HTTP server). Scoring: `scripts/tts_bench.py` (soxr 16k resample, batched whisper round-trip, SIM-o). Metric layer: `lib/tts/`. Chart: `scripts/chart_tts.py`.
- sm_120 notes that cost time: Fish from-source needed `pyaudio` dropped (no PortAudio headers; mic-only, irrelevant to batch synth); torch 2.8.0+cu128 runs the 4B dual-AR + codec on sm_120; the SIM-o path needs a local s3prl cache with a wavlm-only hubconf + torchaudio-2.x shims (the legacy s3prl zoo breaks on `set_audio_backend`/`sox_effects`).
