# WITCHEER gold-standard posts (judge calibration set)

These are operator-approved, *posted* pieces — the ground truth for "what good looks like."
The judge scores new drafts against the rubric AND these examples. They teach taste the rubric
can't fully spell out: lowercase-leaning casual-but-precise register, thesis-first, `~~~`
separators, data as bullets, an explanation of the *mechanism* (not just numbers), a pragmatic
"worth it if / not if" close, emoji optional. "X to Y" for ranges, "x" not "×", no "→".

---

## GOLD 1 — REAP pruning (head-to-head, anti-hype mechanism)

"Expert-pruning shrinks the model, so it must run faster."

I tested that and it doesn't.

REAP takes Qwen3.6-35B-A3B and prunes 20% of its experts down to 28B.
I ran it head-to-head against the unpruned parent.

~~~
what pruning actually changed:

- VRAM: 27.3 to 21.6 GB. a real ~6GB win.
- speed: 260 to 247 tok/s. no gain - slightly slower, even.
- quality: behind on all five (MMLU 94.7 to 87.7, HumanEval 98 to 94, HellaSwag 87 to 82, GSM8K 92 to 90, ARC 97 to 95).

~~~
REAP prunes whole experts, that cuts total params, but the active path per token is still ~3B either way. generation speed tracks active params and memory bandwidth, not total size.

you free VRAM, not time.

~~~
so expert-pruning here is a VRAM play, not a speed play.
it is worth it if you're memory-bound and need the headroom. not worth it if you're chasing tokens-per-second.

(n ~100 per task, so treat each gap as modest - but the parent won every single one.)

---

## GOLD 2 — HumanEval harness self-correction (caught my own bug)

PSA for anyone running local benchmarks: your HumanEval harness is probably lying about reasoning models.

I caught mine doing it. GPT-OSS-120B scored 22% on my HumanEval. it's a frontier coder, so 22% is absurd.

so I pulled the raw model outputs.

the bug was my harness:

- reasoning models think INLINE before writing code. my stop sequence ("\ndef ") fired inside that reasoning and cut generation off before the code even existed.

- and I was calling .strip() on the response, which destroyed the indentation of the code that did come back.

after fix I rerun - same model, same prompts:

GPT-OSS-120B: 22 to 98%.

---

## GOLD 3 — gpt-oss-120B on one 5090 (headline result + honest cost)

OpenAI's GPT-OSS-120B runs on a single RTX 5090.

it's a 59GB model in native MXFP4. it doesn't fit in 32GB of VRAM.
the move is MoE offload: keep attention on the GPU, spill the expert weights to system RAM (llama.cpp --n-cpu-moe).

this way, only 5.1B of 117B params fire per token, so the CPU side stays cheap.

with reasoning on, measured on my box, temperature 0, ~100 items per task (MMLU 114):

- MMLU 89.5
- GSM8K 97.0
- HumanEval 98.0 pass@1
- ARC-Challenge 95.0

that's a good frontier-grade scores, on one consumer GPU.

~~~
it is quite slow tho: 47 tok/s generation.

that's because the experts live in RAM, so token speed waits on the CPU, not the 5090.

prefill is fine with 473 tok/s at 512 ctx. it is generation that pays the offload tax.

the model is usable, not fast. but you get a real frontier model you fully own, on hardware you can buy, for the price of patience.
