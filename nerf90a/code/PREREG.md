# PREREG: "did they nerf Qwen3.8-27B?" same-file audit (idle-queue item 90a-qwen38-nerf-audit)

Frozen 2026-11-06 (CET), before any counted output. The files listed in `FROZEN.sha256` (this file, the harness copy, the item sets, the runner, the verdict rule, the day-0 comparator json) must hash to what is recorded: the item's PLAN phase refuses to run if any differ. Smoke runs use the slug `nerf90a-smoke` and are never counted.

## the claim under test

A r/LocalLLaMA thread of 2026-11-05 (read 2026-11-06 through Arctic Shift, ~1,100 score; most replies read it as satire) says Qwen3.8-27B got worse. We do not name the poster and we do not test hosted-API claims: a hosted endpoint has no file to hash and no run we control, so the clip says so. What we can test is the OPEN file:

1. is the file we benchmarked on 15 Aug the same bytes today, and did the weights repo change?
2. does the same board, run again with the same harness, give the same answers and the same scores?

## the facts re-read 2026-11-06 (Hugging Face API, live in `facts.py` at run time)

- `Qwen/Qwen3.8-27B`: 18 safetensors shards; every shard has the same LFS oid at `main` as at the upload commit 72a217af (13 Aug). Newer commits are README/LICENSE/template edits, the latest a model-card colour fix on 5 Nov. https://huggingface.co/api/models/Qwen/Qwen3.8-27B/commits/main
- `unsloth/Qwen3.8-27B-GGUF`: Q6_K (22,884,408,288 B) oid `562fbf76...6727` and plain Q4_K_M (17,106,775,008 B) oid `7e78da5d...e169` at commit f1bfb127 (15 Aug), the files our day-0 board ran (our download metadata carries the same two oids). The repo removed both names on 19 Aug (commits 24fde402, e1d8a267), so the oid of record is the last listing. An earlier cut of both files (14 Aug 15:17, commit 4604b899) had different oids: day-0 repos re-cut files inside the first hours; the board ran after the re-cut.
- the Q6_K on this box is `~/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf` (the file the always-on server uses). The plain Q4_K_M had been cleaned off the box; it was re-fetched by commit id f1bfb127 on 2026-11-06 and its sha256 equals the recorded oid.

## design (all frozen)

- file A = Q6_K (the headline: it is the file the served model uses), file B = Q4_K_M (secondary).
- build OLD = llama.cpp 5d16e81dd (b10371, 11 Aug): the build the board's config pointed at from 11 Aug (`llama_cpp.server_bin`, vault note second-tier-hc-status.md), i.e. the day-0 build; UNVERIFIED because the day-0 meta.json predates the provenance block. build NEW = llama.cpp 839a572a4 (b11777, 20 Oct): the build the always-on server uses today.
- passes, in this order: NEW+Q6_K (headline), OLD+Q6_K, NEW+Q4_K_M, OLD+Q4_K_M. Server line identical to the board's (`-m F --port P --n-gpu-layers 99 --host 127.0.0.1 --jinja`, no context flag), one client, requests one at a time, think-off through `chat_template_kwargs`, temperature 0, max_tokens 2048 (4096 for HumanEval). Harness = the `lib/evals` tree of benchmark-rig commit 6c7ecd8 (17 Aug, the commit that wired GPQA in; the GPQA loader itself is 4062ac4 the same day, and the other four tasks have no behavioural change between the board's 15 Aug run and this commit: later commits only add counters and fallbacks) copied unchanged to `nerfevals/`: same prompts, few-shot, shuffle seed 42, answer parser. UNVERIFIED: the exact working tree of the 15 Aug run is not recorded in the day-0 meta.json.
- items = the day-0 items: GPQA-diamond 198 (per-item day-0 answers exist), HumanEval 164 (day-0 failing task ids exist), ARC-Challenge 1,172 (day-0 totals only), and MMLU: 19 of the 57 subjects (sorted, every 3rd), each subject's day-0 50% sample (`random.Random(42)`, same indices), 2,614 items (day-0 per-subject counts). Not re-run: GSM8K (day-0 took 62-72 min on its own) and the remaining 38 MMLU subjects and HellaSwag: they do not fit 75 min of card, and the omission is printed on the card.
- identity of the item sets is proven on CPU in the PLAN phase (`nerf_common.identity_check`): per-item expected letters equal the day-0 record for GPQA, per-subject sample sizes equal the day-0 totals for MMLU, ARC and HumanEval sizes equal.

## metrics and rule (applied by `nerf_summary.py`, nothing worded by hand)

- per task: score = correct / n; day-0 band = Wilson 95% interval of the day-0 score (`lib/ci.py` formula); today's own Wilson interval is printed too. A task is OUTSIDE when today's score is below the lower or above the upper edge of the day-0 band, and it counts as MOVED only when it is also significant at p < 0.01: exact McNemar on the discordant items when per-item day-0 data exist (GPQA, HumanEval), two-proportion z test otherwise (ARC, MMLU slice).
- paired per-item agreement (same answer / same pass-fail) is reported for GPQA and HumanEval against day 0, and old-build against new-build for every task that ran in both. Agreement is reported, not gated.
- verdict, headline pass only (NEW+Q6_K), in this order:
  1. INCOMPLETE: any of the four tasks has fewer items than the day-0 total -> exit 3, resumes, no `summary.json`.
  2. CHANGED FILE: the sha256 of the Q6_K on disk differs from the recorded oid.
  3. SHIFTED DOWN / SHIFTED UP / SHIFTED MIXED: at least one task MOVED (all down / all up / both).
  4. SAME FILE, SAME SCORES: otherwise (all four tasks inside the band, or outside without significance).
- attribution (only when SHIFTED): if the OLD+Q6_K pass is complete and itself SAME FILE, SAME SCORES, the sentence is "the difference is the runtime build, not the weights"; if it also shifted, "the difference is not the build alone". With the file hash matching, SHIFTED never means the weights changed.
- line 1 is built by `line1()` from the verdict and the GPQA and HumanEval counts. SAME: "Qwen3.8-27B was not nerfed: the Q6_K file is byte-identical to August and the same board gives the same scores (...)". A SHIFTED result is posted as a shift with its attribution, framed as a measurement of the runtime on one rig. A null here is the expected and valid result: the joke is that the hash cannot lie.
- Q4_K_M and OLD-build passes are secondary: reported with their own words, never gate the headline verdict.

## limits to print with any number

greedy decoding, think-off, one run per pass, one rig; n = 198 and 164 on the paired tasks (one GPQA item is ~0.5 point, gaps of a few points are noise); MMLU is a 19-subject slice, not the full board; OLD build identification is inferred (UNVERIFIED); the Q6_K and Q4_K_M are community GGUFs of the open weights; hosted-API nerf claims are out of scope; llama.cpp builds differ between day 0 and today, which is why both are run.

## render

CPU only on capsule (PIL + the static ffmpeg), 1920x1080 16:9, ~68 s: result card first (8 s), then the hash fingerprint typed against August's oid, then the answer-by-answer replay with a running same-answer counter (letters and pass/fail only, no question text), then the band picture; ends on the card. Mock data is stamped MOCK on every frame. `render_nerf.py` is not frozen (a post-run layout fix is a second commit with a README line).
