# Qwopus3.6-27B-Coder-Compat — the prior tune's two regressions, healed

**Model:** [Jackrong/Qwopus3.6-27B-Coder-Compat-MTP-GGUF](https://huggingface.co/Jackrong/Qwopus3.6-27B-Coder-Compat-MTP-GGUF) — a "compatibility" re-release of [Qwopus3.6-27B-Coder](https://huggingface.co/Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF) (a coder SFT of the Qwen3.6-27B lineage). The release notes claim **less looping across harnesses** and **better thinking stability**, and advise running temp 0.85–1.0.

We benched the original Coder two weeks ago and found two regressions against its own base, Qwen3.6-27B: a **degraded MTP draft head** (1.4–1.6× where the base head does 1.8–2.2×) and **real SWE-bench resolve below base** (17/30 vs 19/30). So the question for this release is narrow and answerable: did the compat work fix the regressions, and at what cost to capability?

Same harness as the base and the prior Coder. Q6_K, think-off, temperature 0 — we hold temp at the board value so the only variable is the model. (The vendor recommends 0.85–1.0; temp 0 is the maximally-adversarial low-temp harness their notes warn about, which makes it the right place to test a "less looping" claim head-to-head against the prior version.)

---

## Leg 1 — Quality: flat

| | MMLU | ARC-C | HellaSwag | HumanEval | GSM8K | **q_avg** |
|---|---:|---:|---:|---:|---:|---:|
| **Coder-Compat** | 87.86 | 96.67 | 95.30 | 90.85 | 97.80 | **93.70** |
| Qwopus-Coder (prior) | 87.5 | 96.8 | 95.2 | 93.3 | 97.5 | **94.1** |
| Qwen3.6-27B (base) | — | — | — | — | — | **94.0** |

Flat — 93.70 against the prior 94.1 and base 94.0, inside the 50%-sample swing. HumanEval is the only axis that moves (90.85 / 149-of-164 vs the prior 93.3), and not enough to read through the noise. A harness-compat release shouldn't move the weights, and it doesn't. The interesting axes are the two it was *supposed* to fix.

## Leg 2 — MTP draft head: recovered to base

Self-speculative decode, draft-depth 2, q8 KV — the same protocol as our base and prior-Coder numbers. "base" here is the Compat model with drafting off.

| workload | base tok/s | +MTP tok/s | **Compat ×** | prior Coder × | base Qwen head × |
|---|---:|---:|---:|---:|---:|
| Q&A | 62.7 | 122.1 | **1.9×** | 1.4× | 1.9× |
| code | 62.7 | 124.5 | **2.0×** | 1.4× | 2.0× |
| prose | 62.8 | 127.0 | **2.0×** | 1.4× | 1.8× |
| JSON | 62.7 | 142.7 | **2.3×** | 1.6× | 2.2× |
| repetitive | 62.6 | 144.0 | **2.3×** | 1.6× | 2.2× |

This is the clearest result. The prior Coder's "natively finetuned" draft head accepted *worse* than the model it was built on — flat at 1.4–1.6× where the base head climbs 1.8→2.2× with output predictability. **Compat's head is back on the base curve: 1.9–2.3×, peaking near 144 tok/s.** Whatever the compat work touched, it went past chat-template plumbing and restored the draft head's acceptance. The "100 tps" the line claims clears comfortably. (Absolute baseline is 62.7 tok/s here vs ~70 in the June-12 run — a llama.cpp build shift, b9653; the *speedup ratio* is the protocol-comparable number, and that's what recovered.)

## Leg 3 — Agentic Score: 100.0, and saturated

| | Agentic Score | tool-eff | stable | tok/task |
|---|---:|---:|---:|---:|
| **Coder-Compat** | **100.0** | 1.00 | 100% | 196 |
| Qwopus-Coder (prior) | 100.0 | 1.00 | 100% | 195 |
| Qwen3.6-27B (base) | 98.6 | 1.00 | — | — |

Perfect score, zero instability across the suite — and the codeact and multistep axes run **thinking-on**, so the "thinking stability" claim holds where we can see it. But this axis is **saturated**: it can't tell Compat from the prior version, both sit at 100.0. On our [agentic leaderboard](https://huggingface.co/datasets/witcheer/agentic-score-leaderboard) this is the recurring "blind-within-family" problem — the synthetic score ranks across the zoo but not between siblings of one base. The real discriminator is the next leg.

## Leg 4 — Real SWE-bench: recovered to base parity

Same 12 SWE-bench Verified bugs, real repo tools in Docker, graded by the official harness. Base and prior were re-graded on the identical 12 from their cached generations, so all three are one protocol.

| | synthetic agentic | **real SWE-12 resolve** | give-ups (empty patch) |
|---|---:|---:|---:|
| **Coder-Compat** | 100.0 | **8/12 (67%)** | 3 |
| Qwen3.6-27B (base) | 98.6 | **8/12 (67%)** | 3 |
| Qwopus-Coder (prior) | 100.0 | **7/12 (58%)** | 4 |

**Compat resolves the same eight bugs as base, exactly.** The prior Coder missed one of them — `pytest-dev__pytest-6202` — and gave up (empty patch) one more time than base. Compat recovers that bug and lands on base's resolved set with one fewer give-up. The three none of them solve (`seaborn-3187`, `requests-1921`, `sphinx-8621`) are stable across all three; Compat also patched `pylint-7080` but the patch failed its tests.

On the full 30-bug set the prior Coder scored 17/30 (57%) against base's 19/30 (63%); the 12-subset reproduces that gap (7/12 vs 8/12) and shows Compat closing it. The +1 is within noise on n=12 — so the honest read is **recovered to base parity, beat the prior tune**, not "now better than base."

---

## Verdict

A compatibility release that does exactly what it says, measured. The prior Qwopus-Coder carried two regressions against its own base — a broken draft head and below-base real-bug resolve. **Compat heals both back to base parity and keeps quality and synthetic-agentic flat.** No capability traded for the fix.

**Worth it if** you wanted the Coder tune's agentic profile but were paying the draft-head tax — the speed penalty is gone and real bug-fixing is back to base. **Not a reason to switch** off the base Qwen3.6-27B for bug-fixing alone: on these 12 bugs they resolve the identical set. The win is that the coder tune no longer costs you anything to run.

---

*RTX 5090 32GB · llama.cpp b9653 · Q6_K · think-off · temp 0 · 50% quality sample. Quality + speed via `run_treatment.sh`; MTP via `scripts/bench_mtp_workload.sh` (draft-depth 2, q8 KV); agentic via `gate_and_run.sh`; SWE-bench via `lib/agentic/native/run_swebench.py` on the 12-bug subset, graded by the official `swebench` harness. Base and prior Coder re-graded on the identical 12 from cached generations.*
