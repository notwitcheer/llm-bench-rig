# item 93: Basalt v0.3.0 against llama.cpp on the identical Qwen3.8-Flash-Next GSQ-RCO Q2_0 weights

STAGED 2026-11-06, ran 2026-11-07 (result below). Tests the Basalt claim (r/LocalLLaMA 1xomy0i: about 3.5x llama.cpp decode without drafting, about 5x with it, RTX 5080 16 GB) on one RTX 5090 + 59 GB RAM.
`PREREG.md` is frozen by `FROZEN.sha256`; `93-basalt-race.sh` is the capsule queue item (night-lib contract); `dl_verify.py` is the size-checked, free-space-guarded HF download wrapper. Harness files are the item-89 (`../ripple89/`) ones with a Basalt flavour.
Staging smokes are not counted.

## result (run 2026-11-07, Sat 01:21-02:13 CET, one RTX 5090 32 GB + 59 GB RAM)

verdict on the pre-registered rule: **BASALT AHEAD**.

- decode, no drafting (9 clean runs each, 700 tokens): llama.cpp 89.1 tok/s, Basalt 259.0 tok/s (2.91x)
- decode, each engine's own drafting: llama.cpp n-gram 98.3 tok/s, Basalt MTP 500.4 tok/s (5.09x)
- prefill 16K / 48K: llama.cpp 1,409 / 1,398 tok/s, Basalt 7,799 / 9,691 tok/s
- the same Hermes Agent task (fix a small repo with 4 failing tests, tests untouched, no drafting, 2 takes per engine, all pass): Basalt 12.6 s and 12.3 s, llama.cpp 38.3 s and 39.6 s
- the guide we tested (RTX 5080 16 GB, another quant) quoted about 3.5x without drafting and about 5x with it

deviation, stated before any post: PREREG.md lists the llama.cpp `--n-cpu-moe` ladder as 12, 13, 14, 16; the frozen script ran 10, 11, 12, 14, 16 and picked 11 (90.0 tok/s). this cuts in llama.cpp's favour (12 would have given 86.6) and the verdict holds either way. PREREG.md is unchanged.
