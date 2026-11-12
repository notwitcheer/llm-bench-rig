# item 99z pre-registration: does the FP8 (f8_e4m3) KV cache change our Hermes task result vs f16 KV, same model file, one RTX 5090

frozen 2026-11-12 (Thu), before any counted output. no card time used before this freeze.

## question

Does a llama.cpp FP8 KV cache (f8_e4m3, merged via PR 28392 + 28413) change the number of our Hermes Agent tasks passed on one RTX 5090, for the same model file, same build, same flags, same prompt? This is a quality check of a VRAM saving, not a speed test.

## design (frozen)

- model: Qwen3.8-27B Q6_K, ~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf, 22,884,408,288 bytes (on disk since 2026-08-14, size-checked against HF before this freeze). no download in this item.
- engine: llama.cpp master pinned 09338b85e8b230742fa56f93a1d9f0bf006ca2d3 (PINNED_SHA; the tarball has no .git, so --version prints commit unknown), source from mercury tarball, CUDA sm_120, nvcc 12.8, gcc-14. both FP8 PRs confirmed live 2026-11-12 (GitHub API): 28392 merged 2026-11-10 as 202c1a229, 28413 merged 2026-11-11 as 9e9619d03, both ancestors of the pin (ahead 47 and 37, behind 0). the cache-type spelling was read from the built binary's --help: -ctk/-ctv allow f8_e4m3.
- arms (only the KV type differs): f16 = default cache; fp8 = `-ctk f8_e4m3 -ctv f8_e4m3`.
- measures (brief, frozen): VRAM used at the same context (nvidia-smi after load), max context that fits (short sweep, both arms), decode tok/s from server timings, Hermes task passes. the headline leg is the pass test; the extension leg is a long-context leg at the max the f16 arm cannot fit.
- other flags: -ngl 99, flash-attn on, --jinja, ctx 65,536, one slot, 6 threads, --reasoning-budget 8192 with the series budget message.
- agent: Hermes pin 9bda188, own HERMES_HOME per take, toolsets terminal + file, approvals off, memory off, max 60 turns, wall cap 10 min per take.
- sampling: temperature 0.6, top-p 0.95, top-k 20, seed 721, identical on both arms.
- task: prompts/fix-shopcart.txt on the seed repo (4 planted bugs, 5 tests, 1 passing and 4 failing on the untouched seed).
- order: headline 3 takes f16, then 3 takes fp8 (one server load per arm). extension legs: +5 takes per arm, then the long-context leg.
- reported, never gated: VRAM at the same ctx for both arms, decode tok/s, time to done, tool calls.

## frozen hashes (the item refuses to run when any differs)

- wf_summary.py (verdict rule + self-test): __H_SUMMARY__
- rp_gate.py (pass test, unchanged): __H_GATE__
- prompts/fix-shopcart.txt: __H_PROMPT__
- seed repo (find|sort|sha256sum|sha256sum over seed/, pycache excluded): __H_SEED__
- mc_mkhome.py: __H_MKHOME__

## rule (wf_summary.py applies it as written; --selftest proves eight branches)

- take CLEAN: ended done or wall, at least 3 tool calls, request errors at most 20%.
- take PASS: CLEAN and gate PASS (5 passed, 0 failed, 0 errors; tests/ untouched; no harness-dir access).
- arm BROKEN: no load, failed smoke, or no CLEAN take.
- TIE: same pass count. CHANGED: differs by 1, or fewer than 8 takes per arm (no direction). NEW MORE/FEWER PASSES: differs by >= 2 with >= 8 takes per arm (names a direction; here the direction is fp8 vs f16).
- decode tok/s is reported, never gating.

## how a null or a loss is framed

TIE: "fp8 KV did not change the result on our task" (a result). CHANGED: "the pass count moved, n too small for a direction". FEWER: a measured quality cost of FP8 KV on this task, reported with the f16 arm as reference. BROKEN: suspect flags and harness first; internal only.

## limits

one model file, one card, one task shape (4 small planted bugs), 3 takes per arm headline and up to 8 per arm in extension legs, pass counts are noisy (3 takes cannot separate 60% from 70%), one llama.cpp master commit (09338b85, not a release), no speed claim.
