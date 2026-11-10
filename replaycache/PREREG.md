# PREREG: replay-cache (idle-queue item 96r-replay-cache)

Frozen 2026-11-09 (CET), before any counted output. This file, the code and the selection are listed in `FROZEN.sha256`; the item's PLAN phase refuses to run if any of them changed. Runs with the slug `-smoke` and everything under `run/mock*` are never counted.

## the question

NVIDIA's CPU-ARC paper (arXiv 2611.09891v1, submitted 2026-11-06) says a cache of compiled reasoning, replayed on a CPU with a small model pulling out the numbers, matches the model that produced the reasoning at a fraction of the cost. Does that hold on questions the cache was NOT built from, and what does the same cache do when a question looks like a cached one but carries one extra clause (a near-miss)?

Three things the paper's headline does not measure, which this run does:
1. held-out accuracy. In the paper each cache entry is built from one test problem's own transcript and judged against the golden label before it is registered, so the reported cache accuracy is accuracy on the problems the cache was built from. Here an entry is built from ONE instance per template, the answer key is kept out of the build loop, and the arms are scored on other instances.
2. an independent answer key. Apple's own labels (below), not a model-written one.
3. the near-miss axis (what share of new queries hit the cache, and how often a hit returns a confident wrong answer) and a target-alone control (the same small model with no cache).

## data (primary source: HF dataset apple/GSM-Symbolic, ungated, CC-BY-NC-ND 4.0, eval only)

- `main/test.jsonl` 5,000 rows = 100 templates (`original_id`) x 50 instances; `p1/test.jsonl` 5,000 rows = the same 100 templates x 50, each with one added clause. Final answer = the number after `####` in the `answer` field. File sha256 values are inside the frozen selection.
- We publish item ids and our own outputs, never a copy or modified copy of the set.
- Selection (`sel.json`, frozen): `random.Random(20261109)`; 20 templates drawn from the templates present in both splits (sorted ids); per template 10 held-out instances from `main` (instances 1 to 49; instance 0 is the build instance) and 10 near-miss instances from `p1` (instances 0 to 49). n = 200 held-out queries and 200 near-miss queries. Items run in a fixed interleaved order (round-robin across templates).

## arms

- A, source from scratch: Qwen3.8-27B Q6_K on llama-server (GPU, own port, never :8090), reasoning on, `--reasoning-budget 16384` with the series budget message, house sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0), 4 slots, one completion per question, max 20,000 tokens. The prompt asks for the final answer on the last line as `#### <number>`. The parser takes the number after the last `####`; with no marker it takes the last number in the answer; no number = wrong. A number is right when it is within 0.005 of Apple's label.
- B, cache replay, CPU only (`CUDA_VISIBLE_DEVICES` empty, llama-server `-ngl 0`): embed the query, cosine against every entry's text (description + build question), take the best entry and its score `s`; the stock 1B (Llama-3.2-1B-Instruct Q8_0, bartowski GGUF, NOT fine-tuned) is asked for the entry's schema as a JSON object (schema-constrained decoding, temperature 0); `solve()` runs in an isolated child process. A crash, a missing param or a non-number is a miss (the cache serves nothing for that query). The 1B extraction runs for the best entry of EVERY query so any threshold can be scored afterwards.
- C, the 1B alone, CPU only, same questions, no cache: C = brief reasoning then `####` (700 tokens max, temperature 0), Cd = the number only (24 tokens max). The stronger of the two on held-out main is the control the rule uses (conservative for the cache).

## build (GPU, house model as source and builder)

For each template the source solves the build instance (instance 0 of main); the builder gets the question plus the full transcript and writes ONE Python function `solve(params: dict) -> number`, a parameter schema (list of name, description, example), a one-line description of the question family and the params of that instance. Validation loop, at most 3 rewrites: the function, run on the builder's own params for the build instance, must reproduce the SOURCE's final answer (not Apple's label: the answer key stays out of the build loop, which is the deviation that fixes the first caveat), and its result must change when a param changes (no hard-coded answer). The label of the build instance is logged for the report only. An entry is stored as `entries/<template>.py` + `.json`. A template with no validated entry after 4 attempts stays without an entry (its queries route to the other entries, which is a real cache with a gap).

## named deviations from the paper


house model Qwen3.8-27B instead of Claude Opus 4.5 as source and builder; the paper's compiled form (section 2.2.3: a plain module plus small-model calls) without the semantic-function network; a stock 1B instead of the paper's fine-tuned 1B; our embedder (EmbeddingGemma 300M Q8_0 GGUF from ggml-org, run on CPU, brute-force cosine) instead of their FAISS setup; entries built from one instance per template, not from each test problem; GSM-Symbolic instead of their eight benchmarks; 20 templates, one run, one seed list. The thinking-budget flag and message follow our other local runs (`--reasoning-budget 16384`); the message text is reworded for a task with no tools ("Stop planning and give the final answer now in the requested form."). The 27B runs four slots at once (no speculative decoding), so its seconds per question include the four-way batching.

## routing and extraction (as built)

Best entry only (no top-k). The embedder text is `title: none | text: <description>\n<build question>` for an entry and `task: search result | query: <question>` for a query. The 1B sees the entry's description, the build question with its own params as a worked example, the schema's parameter meanings and the new question, and answers through a JSON-schema constrained decode (all schema keys required, numbers only, temperature 0, seed 1, prompt cache off). `solve()` runs in a child process with a 6 s CPU limit, 1 GiB memory limit and no imports other than math and fractions. Entry code with imports other than math or fractions, `open`, `eval`, `exec` or `getattr` is rejected at build time and counts as a failed attempt.

## threshold

Set on held-out main only: `tau` = the `s` of the ceil(0.9 x 200) = 180th highest-scoring SERVED held-out query, i.e. the strictest threshold at which at least 90% of held-out main are cache hits (a looser threshold could only raise the near-miss hit rate, so this choice cannot manufacture the trap). A hit = served (a number came out) and `s >= tau`. Then `tau` is applied unchanged to p1. If fewer than 180 held-out queries are served, `tau` is the lowest served score and the card says the 90% floor was unreachable.

## metrics and rule

Per arm on held-out main and on near-miss p1: accuracy with Wilson 95% intervals, hit rate, cache hits that are wrong, CPU wall per query (embed + extraction + solve), 1B tokens per query, source tokens and wall per query, card minutes, and the paired A vs B discordant counts with the exact two-sided McNemar test (both the gated B and the cache-only B).

B has two accountings: B gated = the paper's gated serve (cache first, a miss falls back to arm A's answer for that item); B cache-only = a miss counts as wrong. THE RULE USES B GATED; cache-only is reported beside it.

- CACHE HOLDS if B gated >= A - 3 points on held-out main AND B gated >= C + 20 points on held-out main (C = the stronger 1B control).
- NEAR-MISS TRAP if, at that threshold, at least 20% of the 200 p1 queries (40 or more) are cache hits that return a wrong answer.
- both can fire; the verdict words, applied by `rc_summary.py` in this order, nothing worded by hand:
  1. BROKEN: the build finished (every template has an entry or a recorded failure) with fewer than 10 validated entries.
  2. INCOMPLETE: any of the 400 queries is missing from arm A, B or C (exit 3, resumes, no summary.json).
  3. BROKEN: arm A scores under 50% on held-out main (the harness or the prompt, not the cache method, is the finding).
  4. HOLDS AND TRAPS / HOLDS, NO TRAP / DOES NOT HOLD, TRAPS / DOES NOT HOLD, NO TRAP.
- A null or a loss is a valid result. The subject of a failure is the cache method (or the model), never the harness it ran in. If the cache does not hold, line 1 says so plainly with the three numbers; if it holds and the trap does not fire, line 1 says that. No speed multiplier is claimed anywhere; the card prints measured seconds per question and what produced them.

## line 1 (rule-built, `rc_summary.py` `line1()`)

Four sentences, one per verdict, each carrying [A], [B], [C] and [X] filled from the counts: A = arm A held-out accuracy, B = B gated held-out accuracy, C = the stronger 1B control, X = cache-hit-and-wrong share of the 200 p1 queries. The opening is "NVIDIA says a CPU can replay a big model's reasoning from a cache. I rebuilt it on my 5090", followed by the verdict clause with those numbers.

## what is frozen

`FROZEN.sha256` lists this file, the selection, the server regime, and every script that touches a number: `rc_common.py` (prompts, parser, statistics, sandbox), `rc_gpu.py`, `rc_cpu.py`, `rc_summary.py` (the rule), `rc_chain.sh`, `rc_select.py`, `rc_selftest.py`, `rc_mockrun.py`. The renderer (`rc_render.py`) and the item script are not frozen: they only present or schedule what the frozen files produce.

## development before the freeze (disclosed)

Before this file was frozen, the build prompt, the extraction prompt, the parser and the harness were developed on CPU mock data and on a staging smoke of the real models that used only the selection's first three templates (320, 103, 873) and 30 of their queries. That smoke showed the first build prompt let the builder bake one instance's coincidences into the function for template 320 (a "1/2 of the total left" story where the part that left equals the part that stayed), which broke 6 of 10 held-out queries of that template. The build prompt was then given one generic paragraph telling the builder to work out what each quantity means from the wording so that solve() holds for other fractions, numbers and orderings. No template-specific text was added and the paragraph is in `BUILD_PROMPT` in `rc_common.py` (frozen). Consequence to keep in mind: templates 320, 103 and 873 were seen once in staging, the other 17 templates were not run through the real models at all, and the 10 held-out queries of template 320 informed the prompt change. The staging smoke outputs are never counted and live in `run/smoke*` and `run/replaycache96-smoke`.

## limits to print with any number

20 templates of one benchmark family, one run; Q6_K quantisation, sampling is the server default; stock 1B, our embedder, compiled form only; the entries are built from the same model that is arm A, so A and B are not independent; Apple's labels are the key for scoring, and the source's own answer on the build instance is what the build loop is checked against; GPU seconds are measured with four questions running at once on one slot pool, CPU seconds on six threads of a busy workstation CPU. The playback speed of the left side of the clip is a rendering choice, not a measurement.
