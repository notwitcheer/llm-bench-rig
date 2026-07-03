# Making music on a gaming GPU: ACE-Step 1.5 on one RTX 5090

**A full 4-minute song in under 2 seconds, on a consumer graphics card.** ACE-Step 1.5 is an open-weights text-to-music model. Point it at one RTX 5090 (the current top gaming GPU) and it writes a complete 4-minute track in about 1.75 seconds of compute. That is roughly what the model's makers report for a datacenter A100, and about 6x faster than the last-gen RTX 3090 they also cite. These are the first numbers for this model on consumer Blackwell (sm_120).

The point of the benchmark is simple: you do not need a data center to make music with AI. A card you can buy for a gaming PC does it faster than the song plays.

## The numbers

Two model sizes, both the fast "turbo" variant (8 diffusion steps). Each cell is 6 prompts across different genres, one seed, warm-up discarded.

| model | 30s song | 2-min song | 4-min song | vs real-time (4-min) | peak VRAM |
|---|---|---|---|---|---|
| **2B turbo** | 0.37s | 0.92s | **1.75s** | **137x faster** | 9.4 GB |
| **XL 4B turbo** (higher quality) | 0.58s | 1.43s | **2.9s** | **83x faster** | 14.8 GB |

Times are generation compute (diffusion + audio decode), the same basis as the vendor's own figures. Writing the file to disk adds about 0.2 to 0.4s. Real-time factor (RTF) is song length divided by generation time: 137x means the 4-minute song is written 137 times faster than you could listen to it.

**Where the 5090 lands.** ACE-Step's team reports the 2B turbo at roughly 1 to 2 seconds per 4-minute song on an A100 80GB, and under 10 seconds on an RTX 3090. The 5090 comes in at 1.75s: level with the datacenter card, and far ahead of the 3090. And this is the plain out-of-box path (bf16, no torch.compile, no quantization), so it is a floor, not a ceiling.

## Three things worth knowing

**1. The fast tier is A100-class on a gaming card.** 1.75s for a 4-minute song is the headline. A card built for games keeps pace with a card built for data centers, on a model anyone can download.

**2. Quality costs time, and not much of it.** The XL 4B model is the higher-quality tier. It takes about 1.65x longer than the 2B (2.9s vs 1.75s for a 4-minute song) and about 60% more memory. Still under 3 seconds, still on a single card. Listen to the paired samples and decide whether your ear wants the bigger model.

**3. Longer songs are proportionally cheaper.** RTF climbs from 81x at 30 seconds to 137x at 4 minutes for the 2B model. Because the step count is fixed at 8 regardless of length, the fixed overhead spreads thinner over a longer track. A 4-minute song is not 8x the work of a 30-second one; it is closer to 5x.

## What we did not measure

Speed is the finding here. Audio *quality* is not scored: "which song sounds better" is a subjective call, and a made-up number would not help. Instead the run saves the actual songs, and the paired 2B-vs-XL clips are attached so you can judge by ear. A quality-alignment score (does the audio match the prompt) is the natural follow-up.

The run is DiT-only: the diffusion model generates directly from a text caption, with no planning language model in front. That matches how the vendor measured the speed claim, and it is the path most people will use.

## Config

DiT-only, bf16, SDPA attention (flash-attn not required on sm_120), 8-step turbo, guidance off (turbo has no CFG), batch size 1, seed 0, 48 kHz stereo. RTX 5090 32GB, torch 2.10.0+cu128, ACE-Step 1.5 (v0.1.8). No torch.compile, no quantization. Donald (the box's resident model server) drained for the GPU window and restored after.

## Reproduce

```
# capsule (RTX 5090), in the ACE-Step 1.5 clone + its uv env:
ACESTEP_INIT_LLM=false .venv/bin/python ace_synth.py \
  --model-tier 2b --config-path acestep-v15-turbo \
  --prompts prompts.json --durations 30,120,240 --steps 8 --seed 0 \
  --out-dir ~/ace-out --synth-json ~/ace-out/synth-2b.json --save-audio
# (repeat with --model-tier xl --config-path acestep-v15-xl-turbo)

# Mac: aggregate + chart
python3 -m scripts.ace_bench --synth results/ace_step/synth-2b.json results/ace_step/synth-xl.json \
  --out results/ace_step/ace-step-music.json
python3 scripts/chart_ace.py
```

Prompt set (`dataset/ace_step/prompts.json`): pop, lo-fi, orchestral, EDM, acoustic, hip-hop. Metric helpers `lib/ace_step/` (RTF, aggregation) are unit-tested.

*Model: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (Apache-2.0), paper [arXiv 2602.00744](https://arxiv.org/abs/2602.00744). First RTX-5090/sm_120 figures.*
