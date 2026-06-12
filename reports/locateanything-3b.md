# LocateAnything-3B outside the H100: PBD's 2x survives a consumer card — but the official eval protocol doesn't

**Rig:** one RTX 5090 32GB · transformers 4.57.1 (bf16, SDPA fallback, no MagiAttention/flash-attn) · greedy, temp 0
**Model:** [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) (NVIDIA non-commercial license, 3.83B on disk: MoonViT-SO-400M + Qwen2.5-3B + untied head) — one generalist grounding model for open-vocab detection, referring expressions, pointing, GUI/scene-text/document grounding. Novelty: **Parallel Box Decoding** (PBD) — boxes decode as atomic 6-token blocks in one parallel step instead of token-by-token.

## 1. PBD verified: ~2.1x, and it survives the SDPA fallback

The claimed "up to 2.5x" over AR decoding was measured by NVIDIA with their optimized stack. On consumer
Blackwell (sm_120, CUDA 12.8) neither MagiAttention (requires CUDA ≥ 13) nor flash-attn is available, so
the model runs its SDPA fallback — and PBD still delivers:

| generation_mode | tok/s | forward steps | boxes/s (model stat) |
|---|---|---|---|
| slow (AR) | 99.9 | 56 | 14.3 |
| fast (PBD) | 207.3 | 16 | 31.3 |
| hybrid | 207.2 | 16 | 31.3 |

**2.07x token throughput, 2.19x boxes/sec** on an 8-box detection prompt, deterministic at temp 0.
Inference footprint on normal images: **7.75 GB** — it ran beside a live 27B llama-server.
Two quality notes: `fast` emitted one fewer box than `slow` on the same greedy prompt (speed is not
entirely free), and single-instance grounding cannot abstain — ask for "the leftmost animal" on an
animal-free street and it returns a confident whole-image box. Detection mode's explicit
`<box>None</box>` negatives, by contrast, are clean.

## 2. The wall: the official protocol cannot run on consumer GPUs

The shipped processor allows **25,600 ViT patches** per image (native-res screenshots). Without
flash-attn, MoonViT's SDPA fallback materializes full attention — on a 6016x3384 screenshot that is a
**single 40.6 GB allocation**. An H100 80GB absorbs it; no consumer card can. The largest
`in_token_limit` that fits 32 GB is **12288** (peak 29.5-29.8 GB, GPU otherwise empty), which the
processor satisfies by extra-downscaling big screenshots. Every consumer-GPU number for this model is
therefore measured under forced downscaling — ours included, below.

## 3. ScreenSpot-Pro: 55.3% measured vs 60.3 claimed — and where it actually breaks

Full benchmark, 1,581 instructions, official pointing prompt (`Point to: {instruction}.`), point-in-box
metric, greedy, hybrid mode, 2.9 s/item (75.6 min total), zero OOMs.

| cut | acc | n |
|---|---|---|
| **overall** | **55.3%** | 1581 |
| paper claim (H100, native res) | 60.3 | — |
| text targets | 63.2% | 977 |
| icon targets | 42.7% | 604 |
| 1440p-2.5K screenshots | 58.0% | 849 |
| 4K screenshots | 53.3% | 632 |
| above 4K | 48.1% | 81 |

**The claim is plausible, not refuted.** Our deviation (12288 vs 25600 patches) only pushes accuracy
down, and the resolution gradient (58.0 → 53.3 → 48.1) is consistent with downscale severity — though
bigger screenshots may also simply be denser UIs; the two can't be fully disentangled from this run.
The honest consumer-card number for this model on ScreenSpot-Pro is **~55%**.

**The real fault line is text vs icon (63.2 vs 42.7).** It reads textual targets well and struggles
with abstract iconography — the per-app table says the same: Word 82.1%, EViews 80.0%, Unreal 77.1%
at the top; AutoCAD 26.5%, FruityLoops 38.6% at the bottom (icon-dense, idiosyncratic UIs).

## Honest caveats

- `in_token_limit` 25600 → 12288 is a forced protocol deviation (32 GB ceiling); our 55.3 understates
  what the model does at native resolution.
- Greedy decoding (their reference config samples at temp 0.7); deterministic but not their exact recipe.
- ScreenSpot-Pro metric here is point-in-target-box on the English instruction, one attempt, no retries.
- The model is **non-commercial** (research/evaluation only) — relevant before anyone builds on it.
- PBD speed was measured on one detection prompt shape; the 2.07x is workload-typical, not a sweep.

## Reproduce

`scripts/la_probe.py` (smoke / PBD speed A/B) · `scripts/la_oomtest.py` (the in_token_limit wall) ·
`scripts/la_screenspot.py --token-limit 12288` (full eval) · raw per-item results:
`raw/la-screenspot.jsonl` (this dataset).
