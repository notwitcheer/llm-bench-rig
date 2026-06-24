# Draft-free spec-decode on Qwen3-8B: the win is the workload, and the fancy table buys nothing

**Rig:** one RTX 5090 32GB (sm_120) · Qwen3-8B **bf16** · batch=1, **greedy** · 30 prompts/workload, warm-state (first discarded) · 16.5–16.8GB VRAM
**Method:** one instrumented greedy spec-decode loop; the three arms differ *only* by the drafter — **AR** (no draft), **PLD** (single-line prompt-lookup, the dumb baseline), **Cacheback** (the paper's dynamic LRU n-gram table). No second model, no training, no extra VRAM. Three workloads spanning the repetition spectrum: **code** (HumanEval), **copyctx** (summarize a CNN/DailyMail article), **chat** (open-ended Q&A).
**Metric:** MAT = mean accepted tokens per target forward pass (the portable, implementation-independent number) and realized speedup vs AR.

## The numbers (Qwen3-8B, RTX 5090, 30 prompts each)

| workload | AR | PLD (dumb) | Cacheback (LRU table) | MAT PLD → CB |
|---|---|---|---|---|
| code | 1.00× | **1.45×** | **1.47×** | 1.45 → 1.46 |
| copyctx | 1.00× | 1.30× | 1.30× | 1.30 → 1.30 |
| chat | 1.00× | 1.26× | 1.26× | 1.25 → 1.25 |

Two findings, one of them a clean null.

## Finding 1 — the speedup is real but workload-shaped, and not where you'd guess

Cache-only spec-decode needs no draft model: it proposes the next few tokens by looking up where the recent context appeared earlier in the *same* stream. So its payoff is pure n-gram repetition, and that varies by task:

- **Code is the sweet spot (1.45–1.47×), not summarization.** HumanEval completions are saturated with repeated structure — identifiers, `self.`, `return`, indentation runs — so the lookup hits often (MAT 1.45).
- **Summarization (copyctx) lands in the middle (1.30×), not on top.** The intuition is "summaries copy the source, lookup wins big." But a *three-sentence* summary paraphrases — it doesn't echo long verbatim spans — so the win is moderate.
- **Even open-ended chat gets 1.26×.** The "low-repetition" workload still has plenty of common n-grams (function words, stock phrases). There is no zero — n-gram repetition is everywhere in natural generation.

Net: a free 1.25–1.47× on single-stream greedy decoding, no model and no VRAM, with the largest gain exactly where local agents spend their time (code).

## Finding 2 — the dynamic LRU table ties dumb prompt-lookup (the null)

The headline question was whether Cacheback's machinery — an LRU n-gram table with multiple stored continuations — beats one-line prompt-lookup. On these workloads, **it doesn't**: identical MAT on chat and copyctx, +0.01 on code. The slope chart's PLD → Cacheback segment is flat.

The mechanism is the reason. With leader-length 1 (match on the last token), "look up the most-recent continuation in an LRU table" and "scan back for the last occurrence and take what followed" are the *same algorithm*. The table's extra bookkeeping changes nothing the greedy verifier accepts. Cacheback's published edge over prompt-lookup comes from the parts this dynamic-only arm omits — a **frozen background text corpus** seeded before generation, plus **tree drafting with tree-attention** — not from the dynamic table itself.

This also corrects the brief that started this run. The widely-cited **"1.86×" is Vicuna-7B with a frozen corpus + tree on an RTX 4090** (Spec-Bench), not a modern 8B. There is no per-workload table and no head-to-head-vs-PLD in the paper; both are measured here for the first time on Qwen3-8B / consumer Blackwell.

## The measurement guard that earned its keep

Greedy spec-decode is lossless by construction — a draft token is accepted only when it equals the target's own argmax. We checked it empirically against a separate pure-AR run and found **46072 / 46080 generated tokens byte-identical**. The 8 that differ are not a decoder bug: every one sits at an **exact bf16 logit tie** (top-2 logits equal to the bit, gap = 0.000). At a tie, greedy argmax is resolved by CUDA reduction order, which changes with the forward-pass tensor shape — so appending a draft can flip a coin the model is genuinely indifferent about, and greedy path-dependence carries it forward. (Re-running the AR step at the same shape flips it too.) That is bf16 greedy non-determinism, a property of the model, not the method. The realized speedup also stays ≤ MAT on every cell, the physical sanity bound (a measured speedup above MAT would mean an instrumentation bug).

## Caveats (read before quoting the numbers)

- **The harness recomputes the full prefix each step (no KV-cache reuse).** That depresses *absolute* tok/s and is why the table reports speedup and MAT, not throughput. MAT is implementation-independent — it counts target forward passes saved — and that saving is exactly what carries over to a KV-cached server. The relative ordering (code > copyctx > chat; PLD ≈ Cacheback) is the result.
- **Single-stream, greedy, batch=1.** Under batching, spec-decode economics change (the verifier competes with other sequences); these numbers are the single-user latency regime.
- **Lossless modulo bf16 ties**, quantified above — not glossed.

## Worth it if / not if

- **Worth it** as a zero-cost latency win for single-stream local generation — especially code — when you can't or won't run a draft model: no second model, no training, no extra VRAM, lossless. Plain prompt-lookup (`prompt_lookup_num_tokens` in HF) already captures essentially all of it.
- **Not worth the extra machinery** of a dynamic LRU table over one-line prompt-lookup — it ties on these workloads. The only reason to build the full Cacheback is its *frozen corpus + tree* arm, which is a different (parked) experiment.

## Repro

- Loop + drafters + invariants (dep-free, tested): `lib/cacheback.py` (`spec_decode_loop`, `pld_propose`, `CachebackDrafter`, `greedy_accept`, exact-cap + EOS truncation). Driver: `scripts/bench_cacheback.py` (`--arm ar|pld|cacheback`). Workloads: `scripts/build_cacheback_workloads.py` → `dataset/cacheback/{code,chat,copyctx}.jsonl`. Aggregate + chart: `scripts/{aggregate,chart}_cacheback.py`.
- Losslessness guard (tie-aware, GPU): `scripts/verify_cacheback_lossless.py` — recomputes the logit gap at every AR-vs-spec divergence and asserts each is a bf16 tie.
- Env note: a fresh `uv` venv with `transformers ≥ 4.51` + torch cu128 (the official Spec-Bench `cacheback` branch pins `transformers==4.37.1`, which can't load Qwen3); HF dataset ids must be namespaced (`openai/openai_humaneval`, `abisee/cnn_dailymail`).
