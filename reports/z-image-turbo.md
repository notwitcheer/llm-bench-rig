# Making pictures on a gaming GPU: Z-Image-Turbo on one RTX 5090

**A 1024x1024 image in about 3 seconds, on a consumer graphics card, out of the box.** Z-Image-Turbo is an open-weights (Apache-2.0) text-to-image model from Alibaba's Tongyi lab: a 6B image transformer with a Qwen3-4B text encoder. Point it at one RTX 5090 and it paints a 1024px image in roughly 3.2 seconds of compute, about 19 images a minute, with no quantization and no torch.compile. These are the first numbers for this model on consumer Blackwell (sm_120).

The interesting part is not just that it is fast. It is *what the "turbo" trick costs and does not cost* once you look closely.

## The numbers

Eight prompts across different categories, one seed, warm-up discarded, 8 steps (the distilled default). All on one RTX 5090 32GB, bf16, SDPA attention.

| resolution | compute time | images / min | peak VRAM |
|---|---|---|---|
| 512x512 | 0.83s | 72 | 20.4 GB |
| **1024x1024** | **3.18s** | **19** | **21.7 GB** |
| 1536x1536 | 9.99s | 6 | 24.6 GB |
| 2048x2048 | 23.36s | 3 | 28.8 GB |

Times are pure generation compute (cuda-synchronized). Saving the PNG adds about 0.2s. Peak VRAM is the true per-image tensor high-water mark (see the measurement note below).

## Three things worth knowing

**1. Few-step distillation is the whole trick, and it is linear.** A normal diffusion model needs roughly 50 denoising steps. Z-Image-Turbo is distilled down to 8. Each step is exactly one forward pass through the transformer, so compute time is a straight line in the step count: at 1024px it is 1.69s at 4 steps, 3.20s at 8, 6.23s at 16. Want it twice as fast? Halve the steps. On portraits, 4 steps looks all but identical to 8, so ~1.7s is on the table for a lot of work.

**2. The hard cases actually work.** Legible text and object counting are the two things diffusion models classically get wrong. This one renders "WITCHEER" on a shop sign with the letters correct, and puts *exactly three* rubber ducks in a row when asked for three. Colours and spatial relations land too (a red teapot next to a blue mug; a cat on a stack of books with the plant on the left). At 8 steps. The saved sample grid is attached so you can check the output against the prompt yourself.

**3. Resolution is the real cost, and it is super-linear.** Doubling the side does not double the time. It roughly quadruples it, because attention scales with the number of pixels squared: 0.83s to 3.18s to 9.99s to 23.36s as you climb 512 to 2048. Stay at 1024 or below and it feels instant; go to 2048 and you are waiting 23 seconds for one frame.

## It fits a 32GB card at every size

The model is about 20GB resident (the 6B transformer plus the Qwen3-4B text encoder plus the VAE, all in bf16). Activations add 0.4GB at 512px and climb to about 8GB at 2048px, for a 28.8GB peak. So even a 4-megapixel image fits a single 32GB card with roughly 4GB to spare. There is no VRAM wall in the tested range, only a time wall.

## A measurement note (this bit is reusable)

The peak-VRAM numbers above are read from `torch.cuda.max_memory_allocated()` with `reset_peak_memory_stats()` before each image, **not** from `nvidia-smi`. The reason: PyTorch's caching allocator holds onto the high-water mark of GPU memory it has ever reserved and does not hand it back between generations. So once a 2048px image has run, `nvidia-smi` reports ~31.7GB used for *every* later image, including a tiny 512px one that really only needs 20GB. A naive before/after `nvidia-smi` read would have reported a flat, wrong 31.7GB across the whole sweep. If you profile per-op GPU memory, use torch's own counter and reset it each iteration, or the allocator's retained cache will quietly flatten your curve.

## What we did not measure

Speed and footprint are the findings. Image *quality* is not scored: prompt-alignment metrics like GenEval need a detector stack (mmcv/mmdet) that has to be built from source for sm_120, and that is a task of its own. So this run ships the actual images instead of a fabricated quality number, the same way the music bench shipped the actual songs. A GenEval pass on Blackwell is the natural follow-up.

## Worth it if

You want a fast, genuinely capable image model running locally on a consumer card. A ~3s feedback loop at 1024px makes iterating on prompts pleasant, batch generation cheap, and the classically-hard cases (readable text, correct counts) are handled rather than fudged. Keep an eye on resolution: it is the one axis that gets expensive fast. And remember this is a bf16, no-compile floor, so `torch.compile` likely buys more still.

## Config

bf16, SDPA attention (flash-attn not required on sm_120), `num_inference_steps=9` (8 DiT forwards, the distilled turbo schedule), `guidance_scale=0.0`, batch size 1, seed 42. RTX 5090 32GB, torch 2.10.0+cu128, diffusers 0.37.1 (ships the `ZImagePipeline` — no source build needed). No torch.compile, no quantization. Donald (the box's resident model server) drained for the GPU window and restored after.

## Reproduce

```
# capsule (RTX 5090), any torch-2.10+cu128 / diffusers-0.37.1 env:
python zimage_synth.py \
  --model /path/to/z-image-turbo --prompts prompts.json \
  --resolutions 512,1024,1536,2048 --steps 8 --step-sweep 4,8,9,16 \
  --sweep-resolution 1024 --seed 42 --out-dir ~/zimage-out \
  --synth-json ~/zimage-out/synth.json --save-images

# Mac: aggregate + chart + montage
PYTHONPATH=. python3 scripts/zimage_bench.py --synth results/zimage/synth.json \
  --out results/zimage/z-image-turbo.json
PYTHONPATH=. python3 scripts/chart_zimage.py
PYTHONPATH=. python3 scripts/montage_zimage.py
```

Prompt set (`dataset/zimage/prompts.json`): portrait, two-object, counting, text, spatial, landscape, complex, illustration. Metric helpers `lib/zimage/` (images/min, aggregation) are unit-tested.

*Model: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Apache-2.0). First RTX-5090/sm_120 figures.*
