# replaycache: a reasoning cache replayed on CPU, tested on new questions and near-misses

Idle-queue item `96r-replay-cache` (staged 2026-11-09, run Tue 2026-11-10 01:10 CET, one RTX 5090 block of 52 card minutes plus a CPU chain). Result below.

Claim tested: NVIDIA's CPU-ARC paper (arXiv 2611.09891v1, 6 Nov 2026) says a cache of compiled reasoning, replayed on a CPU by a small model that only extracts the numbers, matches the model that produced the reasoning. The paper builds each entry from the test problem's own transcript and judges it against the golden label, reports no miss or false-hit rate, and runs no small-model-alone control. This harness tests held-out accuracy, the near-miss axis and the control, on apple/GSM-Symbolic (CC-BY-NC-ND 4.0: eval only; `sel.json` holds item ids only, never question text).

Read `PREREG.md` first: it freezes the question, arms, threshold rule, verdict words and the rule that builds line 1. `FROZEN.sha256` lists every file that touches a number; the item's PLAN phase refuses to run if one differs.

- `sel.json` the frozen selection (20 templates, seed 20261109; 10 held-out main + 10 near-miss p1 instances each)
- `rc_gpu.py` card legs: source solves (arm A) and the build loop (source transcript to a validated Python function + schema, checked against the source's own answer, not the label)
- `rc_cpu.py` CPU legs: embedding + routing, the 1B extraction + `solve()` in a sandboxed child (arm B), the 1B alone (arm C)
- `rc_summary.py` the rule (`--selftest` proves every verdict branch), `rc_render.py` the clip, `rc_chain.sh` the detachable CPU chain
- `96r-replay-cache.sh` the queued item; `rc_smoke*.{sh,py}`, `rc_e2e_mock.sh`, `rc_mock*.py`, `rc_selftest.py` the staging proofs (mock data is stamped MOCK on every frame)

Models: Qwen3.8-27B Q6_K (source and builder, arm A), Llama-3.2-1B-Instruct Q8_0 (bartowski GGUF, stock, not fine-tuned), EmbeddingGemma 300M Q8_0 (ggml-org GGUF), all CPU legs with an empty `CUDA_VISIBLE_DEVICES` and `-ngl 0`.

## result (run Tue 2026-11-10, card 01:10 to 02:01 CET, CPU chain finished 03:46 CET)

Pre-registered verdict, applied by `rc_summary.py` with no hand wording: **DOES NOT HOLD, TRAPS** (`run/summary.json`, `run/summary.txt`, `run/readout.md`). 20 of 20 cache entries validated, 0 failed.

| arm (n=200 each) | held-out main | near-miss (one extra clause) |
|---|---|---|
| A: Qwen3.8-27B from scratch | 193/200, 96.5% [93.0-98.3] | 189/200, 94.5% |
| B: cache, a miss goes to A (gated) | 166/200, 83.0% [77.2-87.6] | 15/200, 7.5% |
| B: cache only (a miss counts wrong) | 146/200, 73.0% [66.5-78.7] | 1/200, 0.5% |
| C: stock Llama-3.2-1B alone | 96/200, 48.0% [41.2-54.9] | 41/200, 20.5% |

- cache hits: 180/200 on held-out main (threshold tau 0.4518, set on held-out main only); 186/200 on near-miss, of which 185 returned a wrong answer (92.5% of the 200 near-miss queries).
- McNemar exact, A vs B gated: held-out 29 vs 2 discordant (p 4.6e-07); near-miss 174 vs 0 (p 8.4e-53).
- rule check: CACHE HOLDS needed B gated >= A - 3 points (83.0% against a 93.5% floor: fails); NEAR-MISS TRAP needed >= 20% wrong cache hits among the near-miss queries (92.5%: fires).
- cost: the cache path answered in a median 1.8 s per query on 6 CPU threads; arm A took 17.1 s mean per question with four slots running at once. No speed multiplier is claimed.
- as run: the queued script differs from `96r-replay-cache.sh` here only in two comment lines at the top (header wording); every frozen file matches `FROZEN.sha256`.

Limits, as in `PREREG.md`: our rebuild and not the paper's code (stock 1B, our embedder, compiled form only), 20 templates of one benchmark, one run, the same model builds the entries and is arm A, the 27B's seconds are measured four-up on one slot pool. GSM-Symbolic is CC-BY-NC-ND 4.0, so the repo ships ids and our outputs only.

