# nerf90a: did they nerf Qwen3.8-27B? (same-file audit)

Queue item `90a-qwen38-nerf-audit.sh` (staged 2026-11-06, ran Fri 2026-11-06 16:40-16:54 CET and Sat 2026-11-07 01:05 CET on one RTX 5090). It answers one question with the file and our own August board:

1. sha256 of the Q6_K and Q4_K_M GGUFs on the rig against the LFS oids Hugging Face listed for the files the August board ran, and the weights repo's shard oids against its upload commit;
2. the same board items (GPQA-diamond 198, HumanEval 164, ARC-Challenge 1,172, 19 of 57 MMLU subjects) re-run with the August harness (`code/nerfevals/` is `lib/evals` of benchmark-rig commit 6c7ecd8, unchanged), greedy, think-off, on two llama.cpp builds, scored against the August per-item records in `day0/` with Wilson bands and exact McNemar.

`code/PREREG.md` is frozen (hashes in `code/FROZEN.sha256`, checked by the item's plan phase). `code/render_nerf.py` is not frozen. Hosted-API claims are out of scope. A null (same file, same scores) is the expected result.

## result (2026-11-07)

Pre-registered verdict: **SAME FILE, SAME SCORES** (`run/summary.json`).

- sha256 of both GGUFs on the rig equals the LFS oids Hugging Face lists for the August files (`run/facts.json`: Q6_K `562fbf76...6727`, Q4_K_M `7e78da5d...e169`, GGUF repo rev `f1bfb127`); 0 of the 18 weight shards in `Qwen/Qwen3.8-27B` changed between the upload commit and `main`.
- Q6_K on today's llama.cpp build, same 4,148 items, greedy, think-off: GPQA-diamond 98/198 (August 97/198), HumanEval 154/164 (155/164), ARC-Challenge 1,132/1,172 (1,133), MMLU 19-subject slice 2,192/2,614 (2,191). all four inside the August Wilson 95% band; paired McNemar p = 1.0 on GPQA and HumanEval. per-item records: `run/results/new-q6k/*.jsonl`; August records: `day0/`.
- what ran and what did not: only the headline pass (new build, Q6_K) ran. the item treated the run as complete once that pass wrote `summary.json`, so the three secondary passes (old build Q6_K, both builds Q4_K_M) never started; the secondary block in the summary is empty. this is a harness bug in the item, not a result.
- one run, one rig. hosted APIs are out of scope: there is no file to hash.

- the clip: `code/render_nerf.py` was edited after the run (CPU re-render from the saved artefacts, no new numbers): two card strings now say only today's build was re-run and drop the Q4_K_M ring from the legend. the run itself used the version in the previous commit; render_nerf.py is not in FROZEN.sha256.
