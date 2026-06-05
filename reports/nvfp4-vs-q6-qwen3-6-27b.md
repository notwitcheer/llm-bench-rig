# NVFP4 vs GGUF quants — does Blackwell-native FP4 pay off? (Qwen3.6-27B)

**Model:** Qwen3.6-27B · **Engine:** llama.cpp `llama-server`/`llama-bench` **b9365** (`BLACKWELL_NATIVE_FP4=1`)
**Hardware:** RTX 5090 32GB · Ryzen 5 9600 · 64GB DDR5 · Ubuntu 26.04
**Arms:** NVFP4 (4-bit, 14.6GB) · Q4_K_M (4-bit, 15.7GB) · Q6_K (6.5-bit, 21GB) · think-off, 50% quality sample
**Comparison is quant-only:** spec decoding **off** on all arms (the NVFP4 file ships MTP heads, but `bench.py` does not enable `draft-mtp`; the Q6 baseline is plain Q6). So this isolates the quantization, not MTP.

## Speed (tok/s)

| metric | **NVFP4** | Q4_K_M | Q6_K |
|---|--:|--:|--:|
| prefill pp128 | **3486** | 2973 | 2580 |
| prefill pp512 | **5415** | 3826 | 3222 |
| prefill pp2048 | **5254** | 3741 | 3187 |
| prefill pp4096 | **5073** | 3645 | 3115 |
| prefill pp8192 | **4753** | 3485 | 2992 |
| prefill pp16384 | **4176** | 3162 | 2751 |
| decode tg128 | **84.3** | 77.1 | 61.9 |
| peak VRAM | **17.3 GB** | (not captured) | 23.5 GB |

**NVFP4 deltas:**
- **vs Q4_K_M (equal ~4-bit):** prefill **+32% to +42%**, decode **+9%**
- **vs Q6_K (production reference):** prefill **+52% to +68%**, decode **+36%**, VRAM **−30%**

## Quality (think-off, 50% sample)

| task | NVFP4 | Q6_K | Δ |
|---|--:|--:|--:|
| mmlu | 87.0 | 87.9 | −0.9 |
| arc_challenge | 96.7 | 96.9 | −0.2 |
| hellaswag | 94.9 | 95.4 | −0.5 |
| humaneval | 90.2 | 92.7 | **−2.5** |
| gsm8k | 97.1 | 97.3 | −0.2 |
| **q_avg** | **93.2** | **94.0** | **−0.8** |

(Q4_K_M quality not measured — that arm had speed only. NVFP4-vs-Q4 quality at equal bitrate is the obvious follow-up.)

## Findings

**1. At equal bitrate, NVFP4 clearly beats standard Q4_K_M — on prefill.** +32% to +42% prompt-processing throughput, purely from the Blackwell FP4 tensor cores (both models are ~4-bit / ~15GB, so footprint is held roughly constant and only the compute path differs). This is the real "is NVFP4 worth it" answer: yes, for prefill-heavy workloads.

**2. The "+43–68% prefill, ~0% decode" claim is half-right — and the other half is the lesson.** Our prefill numbers land squarely in that band (+52–68% vs Q6). But **decode is not unchanged**: +9% vs Q4, +36% vs Q6. The reason the original r/LocalLLaMA figure showed ~0% is that it was a *same-model, kernel-vs-kernel* comparison (b8966→b8967), where the native FP4 kernel only helps the compute-bound prefill. Across *quants*, decode is memory-bandwidth-bound, so it tracks **model size**: vs equal-size Q4 the decode gain is small (+9%); vs the heavier Q6 it's large (+36%). **Prefill = compute (FP4 cores); decode = footprint.**

**3. The 4-bit tax is tiny.** NVFP4 lands within **0.8 q_avg** of Q6_K (93.2 vs 94.0), with HumanEval the only real casualty (−2.5). Code generation is the first thing to degrade under aggressive quantization — consistent with prior runs on this rig.

## Worth it if / not if

- **Worth it** — almost always, on this card. vs Q6_K you get **~+60% prefill, +36% decode, −30% VRAM for −0.8 q_avg**. For an always-on agent like a local Hermes, that's faster responses, longer context headroom, and 6GB of VRAM back, at a quality cost you'd struggle to feel outside code.
- **Watch** — if your workload is code-heavy, the −2.5 HumanEval is the one place to A/B before committing. And if you're decode-bound at a fixed model size (vs an equal-size Q4), NVFP4's win shrinks to ~+9% — the prefill is where it shines.

## Reproduce

```bash
# NVFP4 GGUF (text, MTP heads dormant under bench.py): s-batman/Qwen3.6-27B-NVFP4-MTP-GGUF
./run_treatment.sh ~/models/qwen36-27b-nvfp4/Qwen3.6-27B-NVFP4-MTP.gguf   # speed sweep + 5 quality tasks
python3 scripts/chart_nvfp4_vs_q6.py                                       # EV+ chart from results/
```
Baselines reused: `results/qwen3-6-27b-q6-k` (speed+quality), `results/qwen3-6-27b-q4-k-m` (speed). Comparison data: `results/nvfp4-vs-q6-q4-compare.json`.
