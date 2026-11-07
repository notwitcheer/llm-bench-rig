# item 94 pre-registration: QSA fork of Qwen3.8-27B at 255K context against stock llama.cpp, one RTX 5090

frozen 2026-11-06 (Europe/Paris), before any counted output. GPU use so far: staging smokes only (slugs `smoke*`, never counted): one cold request at each of 0 / 64K / 128K / 255K per arm, a prefix-reuse probe, an agent-shaped continuation.

## question

The fork author (pentaCoxian, README fetched 2026-11-06) states, for an RTX 5090: greedy decode 141.8 tok/s and sampled 142.4 tok/s with 255K input, cold prefill 2,756 tok/s (93 s), peak 24.0 GB VRAM, QSA IQ3_S + DFlash2 drafter (4 drafts); on an RTX 4090 stock llama.cpp with the same weights without QSA does 40.6 tok/s (greedy) at 255K against 114.7. The stock figure on a 5090 is not published. Two questions: (1) what does decode speed do from 0 to 255K tokens of context on our 5090 for the QSA fork and for stock llama.cpp, and what is the ratio at 255K; (2) can a real Hermes Agent session on each engine answer questions about the whole Hermes docs text (about 240K tokens) with known answers, and how fast.

## arms (frozen)

- `qsa`: the fork, repo pentaCoxian/Qwen-QSA-llama.cpp at fe08b16 (2026-11-04), built `-DCMAKE_CUDA_ARCHITECTURES=120`, nvcc 12.8 + gcc-14 (plus `-include algorithm`, a header the fork omits). Files (pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF): IQ3_S-mtp 12,329,762,336 B, DFlash2 IQ4_XS 1,038,313,056 B. Server flags and env exactly the fork's `deploy/qsa27b-24gb/serve.sh` (ctx 262144, q8_0 KV, `-ub 256`, `--cache-ram 0`, 4 drafts, probabilistic draft sampling, no vision).
- `stock`: mainline llama.cpp 18ab62a (2026-11-06, same compiler), the non-QSA ISTA-DASLab Qwen3.8-27B-GSQ-RCO IQ3_S-mtp (12,120,016,960 B; same quantiser, same base model, dense attention), the SAME DFlash2 drafter and the SAME flags (serve.sh without the QSA-only env: LLAMA_QSA_RAW_RING, MTMD_ON_DEMAND, GGML_CUDA_EVICT_MIB). The README's own stock comparison uses the same recipe ("same weights without QSA, same prompts, same drafter").
- one slot, no vision, thinking off for the speed rows.

## rows

- depth ladder: cold prompts of 0 / 64,000 / 128,000 / 255,000 tokens (docs text, byte-identical for both arms, a random session line first so no cache applies; the character count per depth is calibrated once with the first arm's /tokenize and stored), task: write a Python script (the README's own best case is code), 256 max tokens, temperature 1.0 top-p 0.95 top-k 20, seeds 11 and 12. Decode tok/s per depth = sum(predicted_n) / sum(predicted_n / predicted_per_second) over the runs with predicted_n >= 100 (the README drops early stops the same way); runs under 16 tokens are counted as early stops. Prefill tok/s = timings.prompt_per_second of the cold request. At 255K one prose request (seed 11) is added and reported, not used in the ratio.
- early-stop probe: 20 requests (seeds 1-20, temperature 1.0) on a shared ~60K-token code-docs prefix, 64 max tokens; reported = requests that ended under 16 tokens, per arm. The README reports about 7% (27 of 400) and an unknown cause.
- Hermes sessions: one real Hermes Agent session per arm (own HERMES_HOME, minted launcher, provider extra_body temperature 0.7 / top-p 0.95 / top-k 20, thinking off, todo toolset only, compression off, no tools that read files), the docs text carried as two ephemeral prefill messages (the same bytes for both arms), then 24 questions typed one per turn ("what is the default value of `<key>`?", answers are single numbers that occur once in the text, depths 0.24-0.89 of the text). Answer keys live in the harness folder, never in the agent's work dir; the agent has no file or terminal tool. Wall cap 40 min per session. Score = questions whose answer contains the key as a whole number (6.0 = 6). The question set is rebuilt from the fetched docs by qa_build.py and its hash is frozen.
- ctx bar in the clip = engine-reported prompt + cache + predicted tokens of the last request (or the live prompt progress), over 262,144.

## ratios and verdict words (applied as written by c94_summary.py; `--selftest` proves the branches)

- R255 = qsa decode tok/s at 255K / stock decode tok/s at 255K (code task, seeds 11 and 12). THE headline.
- verdict on R255: QSA AHEAD (R255 >= 1.5) | LEVEL (0.8 <= R255 < 1.5) | STOCK AHEAD (R255 < 0.8) | BROKEN (an arm has no clean run at 255K) | INCOMPLETE (a Hermes session did not start; provisional, exit 3, resumes).
- claim check (reported, not a verdict): our QSA decode at 255K against the author's 141.8: HELD within 10% (>= 127.6), ABOVE (> 156.0), BELOW (< 127.6).
- line 1 is built by the rule in c94_summary.py from the arms' numbers and the sessions' score. A tie or a loss is a valid result ("one rig"); the 7% early-stop rate and the two-hop recall caveat are quoted from the README as the author's. Hermes Agent is never the subject of a failure: if a session fails to answer, the subject is the engine or the model.

## reported, never gated

prefill tok/s per depth, draft acceptance, VRAM, GPU watts, time to the first answer (cold 240K prefill) and to the last, per-depth recall, the early-stop count, the README's 4090 numbers beside ours.

## clip plan

~8 s result card first (title = the verdict in words, one column per engine, grey how-measured lines, footer with the claim tested and the limits), then the two engines chewing the same 255K prompt (stacked counters: prefill progress and tok/s, then decode tok/s; a speed pill when the prefill is shown faster than real time), then the two Hermes sessions side by side with the context bar, cut 6 s after the faster one has answered its last question. No path, hostname or account on screen (leak mask).

## limits

n=2 seeds per depth; one task for the ratio; IQ3_S files, not the Q6_K we serve; the stock arm is the dense-attention file of the same quantiser, not the QSA weights run without QSA (that combination does not exist); a 2-day-old third-party fork with 1 star; the fork says nothing else may share the GPU; recall questions are single-fact lookups (the README's two-hop recall 11/16 vs 12/16 is the author's and is not retested); thinking off.

## frozen hashes

FROZEN.sha256 (same folder): this file, c94_summary.py, oai_bench.py, qa_build.py, qa/questions.json, qa/prefill.json, and the key file `qa/keys/keys.json`. The item refuses to run when any differs; it also re-runs qa_build.py into a temp dir and compares.
