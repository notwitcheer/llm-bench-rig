# item 99y pre-registration: is unsloth's Dynamic v3.1 re-cut of Qwen3.8-27B (UD-Q4_K_M) different from the October file on our Hermes task, one RTX 5090

frozen 2026-11-12 (Thu), before any counted output (the only GPU use before this freeze: none. the build, the CPU proofs and the stub runs used no card time).

## question

Does the new unsloth v3.1 UD-Q4_K_M file of Qwen3.8-27B pass a different number of our Hermes Agent tasks than the October UD-Q4_K_M file on one RTX 5090, with the same llama.cpp build, the same flags and the same prompt? The vendor claims "better by 5-10% accuracy" (Reddit post 2026-11-11, body text; UNVERIFIED on our card). This is a test of that claim on one task, not a benchmark of the model.

## design (all frozen)

- files: OLD = ~/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf, 16,464,440,224 bytes, LFS oid 322e194f (the file of our 2026-10 rows; unsloth parent commit 43f522df). NEW = ~/models/qwen38-27b-v31/Qwen3.8-27B-UD-Q4_K_M.gguf, 16,564,259,744 bytes, unsloth/Qwen3.8-27B-GGUF main commit 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 (HF tree re-read 2026-11-12), LFS sha256 1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529 (re-hashed on capsule).
- engine: llama.cpp commit cf34d3113 (the ~/llama.cpp-moecache checkout, `llama-server --version` = 0.9.0-dev build 6 commit cf34d3113), CUDA sm_120, --jinja, -ngl 99, flash-attn on, f16 KV (no -ctk/-ctv), ctx 65,536, one slot, 6 threads, --reasoning-budget 8192 with the series budget message, no speculative decoding.
- agent: Hermes pin 9bda188, own HERMES_HOME per take, toolsets terminal + file, approvals off, memory off, max 60 turns, wall cap 10 min per take.
- sampling: temperature 0.6, top-p 0.95, top-k 20, seed 721, identical on both arms (the arms differ only in the model file).
- task: prompts/fix-shopcart.txt on the seed repo (4 planted bugs, 5 tests, 1 passing and 4 failing on the untouched seed).
- order: headline leg = 3 takes OLD, then 3 takes NEW (one server load per arm). extension legs (resumable): up to 8 takes per arm, same blocks. a second task is named in the staging note and is NOT frozen here.
- reported with every arm, never gated: decode tok/s (server timings), time to done, tool calls, peak VRAM, pass counts of the pristine tests.

## frozen hashes (the item refuses to run when any differs; FROZEN.sha256 lists them, PREREG.md is listed there too)

- wf_summary.py (verdict rule + line 1 + self-test): __H_SUMMARY__
- rp_gate.py (the pass test, item 89's, unchanged): __H_GATE__
- prompts/fix-shopcart.txt: __H_PROMPT__
- seed repo (`find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`, run in seed/, pycache excluded): __H_SEED__
- mc_mkhome.py (sampling + toolsets + approvals of every take): __H_MKHOME__

## rule (wf_summary.py applies it as written; `wf_summary.py --selftest` proves eight branches)

- take CLEAN: ended done or wall, at least 3 tool calls in state.db, request errors at most 20% of requests.
- take PASS: CLEAN and the gate passes (pristine tests 5 passed, 0 failed, 0 errors; no file under the agent's tests/ differs from the pristine one) and the agent's tool calls never named the harness dir.
- arm BROKEN: the server did not load, or its tool-call smoke failed, or none of its takes was CLEAN.
- verdict on pass counts (the pass count is the number of PASS takes among the CLEAN takes of an arm):
  - BROKEN: either arm broken.
  - INCOMPLETE: either arm has fewer than 2 CLEAN takes (provisional, the item exits 3 and resumes).
  - TIE: the two arms have the same pass count.
  - CHANGED: the pass counts differ, and the difference is 1, or the arms have fewer than 8 takes each. no direction is claimed.
  - NEW MORE PASSES / NEW FEWER PASSES: the pass counts differ by 2 or more, and both arms have at least 8 takes run. the only branch that names a direction.
- decode tok/s per arm (token-weighted over CLEAN takes) is reported, never gating.

## how a null or a loss is framed

TIE: "v3.1 did not change it on our task" (a result). CHANGED: "the pass count moved on this task, n too small for a direction". NEW FEWER: a measurement of one file pair on one task, with the old file as the reference; the subject is the model file, never Hermes Agent. BROKEN: suspect our flags and harness first, CPU diagnosis on the saved server log before any rerun, internal only.

## limits

one file pair, one card, one task shape (4 small planted bugs), 3 takes per arm headline and up to 8 per arm in the extension legs, different agent trajectories per take (the pass count is noisy: 3 takes per arm cannot separate two files that pass 60% and 70% of the time), master of the moment for the build (cf34d3113 is not a release), the vendor's accuracy benchmark is not reproduced here.
