# item 99a pre-registration: llama.cpp's MoE expert cache against the best static offload, Ornith-1.5-122B, Hermes Agent, one RTX 5090

frozen 2026-11-08 (Sun), before any counted output (the only GPU use before this freeze is none: the build and the CPU proofs ran without the card; a GPU smoke, if one is run, uses the slug moecache99a-moecache-race-smoke and is never counted).

## question

Does the MoE expert cache merged into llama.cpp on 2026-11-06 to 2026-11-08 (PRs #31233, #30966, #31853, #31497; build a2b301831 = tag b12201) decode a 46 GB MoE model faster than plain `--n-cpu-moe` offload on one 32 GB RTX 5090 with 59 GB of RAM, when the model is driven by real Hermes Agent sessions? Test of the shape of the PR's own claim (its tuner example: a 92 GB card, Qwen3.8-Flash-Next IQ4_NL, 24.2 tok/s at a small cache up to 83 tok/s predicted at a large one); not a reproduction of its numbers (different card, model and file).

## design (all frozen)

- model file: Ornith-1.5-122B-UD-IQ3_XXS.gguf, 49,515,667,712 bytes (unsloth GGUF, already on the box, the file of our 2026-10-25 row).
- engine: llama.cpp commit a2b301831d05b07860de508d229d430207127802 (master 2026-11-08 06:11 UTC), CUDA sm_120 build, --jinja, -ngl 99, flash-attn on, KV q8_0, ctx 65,536, one slot, 6 threads, --reasoning-budget 8192 with the series budget message, no speculative decoding.
- arm plain: `--n-cpu-moe 24` (our 2026-10-25 optimum; 26 then 28 only if 24 does not load, the first that loads is used and reported).
- arm cache: `-cmoe --moe-cache-mib N --moe-cache-cpu-share F` where N and F are the "recommended" line of scripts/moe-cache-tune.py at the pinned commit (the PR author's tuner, run unchanged on this card with the cache-arm command line, 6 server starts). If the tuner cannot run or prints no recommendation, N = 16384 and F = 0 and the card says "fallback".
- agent: Hermes pin 9bda188, own HERMES_HOME per take (never the main gateway, never :8090), toolsets terminal + file, approvals off, memory off, max 60 turns, wall cap 10 min per take.
- sampling: temperature 0.6, top-p 0.95, top-k 20 (the Ornith card values as recorded in our 2026-10-25 note), seed 721 (ours), identical on both arms.
- task: prompts/fix-shopcart.txt on the seed repo (item 89's, hashes below): 4 planted bugs, 5 tests, 1 passing and 4 failing on the untouched seed.
- order: plain takes 1-3, then the tuner, then cache takes 1-3. Three takes per arm.
- reported with every arm, never gated: a fixed-prompt probe (temperature 0, 256 tokens, no thinking, second of two runs), time to done, tool calls, GPU power (nvidia-smi 1 Hz, whole card), peak VRAM, pass counts of the pristine tests.

## frozen hashes (the item refuses to run when any differs; FROZEN.sha256 lists them, PREREG.md is listed there too)

- mc_summary.py (verdict rule + line 1 + self-test): 42ad6952882c3e7ac4e023900405b894a5bea9ab2d1743af0f4cff96a2ebbcae
- rp_gate.py (the pass test, item 89's, unchanged): 6b8dcbb5194e4c6773622a1fc48254ba8bc597938c8b8e69661ce881d82b2d57
- prompts/fix-shopcart.txt: d2e53c7d08ad34e5c974a410b6c5fa22d0fae3d0a5779ec530da94ec0a223afb
- seed repo (`find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`, run in seed/, pycache excluded): dc0a36db1200e5203034c020966724b2b2711303fed911b227622c2ee42a5014
- scripts/moe-cache-tune.py at the pinned commit: c3baf2d21ba7119cdd78a941f4f587caee2b376bad1c83bb44d194231ae6998b
- mc_mkhome.py (sampling + toolsets + approvals of every take): d0a70bd714060fbeb79b2e106ad2f40cb7e8ad0144590e9ecc56e590d55252e0

## rule (mc_summary.py applies it as written; `mc_summary.py --selftest` proves twelve branches)

- take CLEAN: ended done or wall, at least 3 tool calls in state.db, request errors at most 20% of requests.
- take PASS: CLEAN and the gate passes (pristine tests 5 passed, 0 failed, 0 errors; no file under the agent's tests/ differs from the pristine one, none added) and the agent's tool calls never named the harness dir. Reported, never gating.
- arm BROKEN: the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
- arm decode = sum(predicted_n) / sum(predicted_n / predicted_per_second) over every request of the arm's CLEAN takes (llama-server's own timings, token-weighted). r = cache decode / plain decode.
- verdict: BROKEN (plain arm broken) | CACHE FAILED (plain fine, cache arm broken) | INCOMPLETE (either arm has fewer than 2 CLEAN takes: provisional, the item exits 3 and resumes) | FASTER (r >= 1.10) | SLOWER (r <= 0.90) | TIE (between).
- line 1 is built by mc_summary.py from the verdict and the two decode numbers: FASTER and SLOWER state both tok/s and the ratio; TIE says "changed ... inside the +-10% tie band". Brackets in the header's MOJO line are filled from summary.json, nothing is worded by hand.

## how a null or a loss is framed

TIE: "no clear speed difference on this card and file" (a result: a 32 GB card with 59 GB of RAM is not the 92 GB card the PR's example ran on). SLOWER: a measurement of one rig, with the tuner's own recommended setting named, and the flag set / engine build as the subject, never Hermes Agent. CACHE FAILED: suspect our flags and harness first, CPU diagnosis on the saved server log before any rerun, internal only.

## limits

one file, one card, one task shape (4 small planted bugs), n=3 takes per arm, different agent trajectories per arm (the ratio mixes decode speed with whatever the model chose to think and write: the +-10% band is wide on purpose; our fixed-prompt depth rows spread at most 0.6%); master of the weekend, not a release candidate of anything we have tested before; the 10-25 baseline of 47.63 tok/s was measured on build b10371 and is quoted only as history, the plain arm is re-measured on the new build; the 5.06 GB of non-expert weights and all KV stay on the GPU in both arms.
