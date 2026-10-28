# agenttest v2: pre-registered design and scoring (lane L1, C1 pass)

written 2026-10-28 14:0x CET (Europe/Paris), BEFORE any model output exists for v2: no v2 task has been run by any model; the checkers were proven first (`results/checker-proof-2026-10-28.json`) and the task set was frozen (`FROZEN.sha256`, `tasks.json`). The queue item re-checks both at every DRYRUN and refuses to run if either changed.

## the question

Which model should a Hermes Agent user run on a local card of 8 / 12 / 16 / 24 / 32 GB? Measured as: real Hermes Agent sessions on 30 everyday offline tasks, one row (model + file) at a time, each row served alone on the card with the context Hermes needs (-c 65536). C1 = the rows that are already on disk. The 5090 stands in for the smaller cards: peak VRAM bytes transfer to other CUDA cards, decode speed does not.

## why a v2 (v1 had a ceiling)

v1 (`agenttest/`, 10 short tasks, 2026-10-22) scored Spark-X2.6-4B Q4_K_M 10/10 and Muse Glimmer 30B 10/10: it cannot separate tiers. v2 keeps the same shape (fresh session per task, deterministic checker) and makes the tasks multi-step: 5 easy (A), 16 mid (B), 9 hard (C). v1's tasks are not reused (a v1 task is an anchor the 4B already solves); the floor of the set is the A group.

## the 30 tasks (`tasks.json`, each with a python checker, hand solution and near miss in `tasks/*.py`)

| cat | ids | what |
|---|---|---|
| files (6) | t01 t02 t03 t04 t05 t06 | exact three-line file, filtered csv total, find a marker line in a 42-file tree, noisy-client log analysis, bulk JSON edit with rules, 40 notes to 3 bullets |
| code (5) | t07 t08 t09 t10 t11 | fix an off-by-one module, fix three bugs in a money module, implement slugify from a 7-rule spec, write a small CLI from a spec, port a python2 script to python3 |
| refactor (5) | t12 t13 t14 t15 t16 | rename a function across a 7-file package, split a module in two and keep old imports, add a required parameter to nine call sites, write tests that catch five hidden mutants, fix broken imports in a package |
| recovery (4) | t17 t18 t19 t20 | two environment faults (script must not be edited), stale lock + missing generated config, a real git merge conflict, make a migration survive bad rows |
| shell (3) | t21 t22 t23 | build a git history with a branch and annotated tag, find the first bad commit, write a shell filter |
| data (3) | t24 t25 t26 | four sqlite questions, csv join with rules, ini to typed json with interpolation |
| long (4) | t27 t28 t29 t30 | build a CLI package from a README spec, merge twelve inbox files in three formats, five questions about a 23-file codebase, repair five bugs behind a 40-test suite |

difficulty by design, not by any model's result: A = one step or one file; B = several steps, one or two files, a hidden second check; C = many files or many steps, or a hidden strict check. Per-task wall budget (agent.run_budget_seconds): A 240-420 s, B 420-720 s, C 720-1200 s (sum 19,920 s = 5.5 h worst case per row, never expected).

All tasks run offline: no web tool is needed (v1's web-fact task is dropped). Every checker derives its expected values from the seed code, never from files the agent can edit; hidden second checks (other inputs, mutants, extra assertions) run after the session. A checker error counts as a fail.

## checker proof (done before any run)

Per task: an empty folder fails, the fresh seed fails, a plausible near miss fails, the hand solution passes with partial score 1.0, the seed is byte-identical on a second build and under a second work root (tree hash). `prove_checkers.py` writes `results/checker-proof-<date>.json`; the item re-runs it on capsule at every DRYRUN (`ALL_OK 30/30` required).

## a run

- Hermes Agent as shipped on capsule (pinned to the install recorded in rows' meta: v0.22.2+2756.g9bda188 (2026.9.24)); one profile per row (`at2<slug>`), never deleted. Config: provider custom -> the row's llama-server, context_length 65536, toolsets hermes-cli, memory off (so no note leaks from one task into the next), approvals at the default (smart, headless = deny), no SOUL edit, `agent.max_turns` default.
- one fresh session per task: `hermes -p <profile> chat -q <prompt> --format stream-json --source tool`, cwd = the freshly seeded task folder, stdin closed. The wall cap per task is `agent.run_budget_seconds` in the profile config (this install rejects `--run-budget` on the CLI). A task is only started with its budget + 120 s left in the block.
- server: llama.cpp b11668 (cd0a5b6a1, the build with spark2_5), `-ngl 99 -fa on -c 65536 --jinja`, f16 KV (the hermes-settings verdict: no setting moved S5 by more than 3 of 100, so the board recipe stays; the VRAM with q8_0 KV is measured in a separate fit leg, accuracy with q8_0 is NOT measured here). Explicit sampling flags per row from the maker card (rows.json: source + fetch date per row), checked against /props before the first task. Hermes sends no sampling fields, so these apply.
- reasoning effort: Hermes sends `reasoning_effort: "medium"` when unset. Per row the item sends one request with medium; if the template rejects it (HTTP 400) it sets `agent.reasoning_effort: high` in that row's profile and records it. That is the only allowed deviation from Hermes' defaults, and it is a template fact, not a Hermes bug.
- thinking is not capped (Hermes as shipped): a runaway shows up as a timeout and counts as a fail.

## scoring (fixed now)

Per task: `pass` = the checker returns true (hard conditions + all hidden checks). `frac` (0..1) is a secondary partial score that is never used in a headline or a pick. Counted per task and separately per row: tool calls; tool errors (non-zero exit, error status, malformed result) NOT caused by an approval block; approval blocks (Hermes' headless "BLOCKED" verdicts); malformed tool calls (unknown tool name, invalid arguments, a raw tool-call string in the final text); wall seconds (process start to exit, includes Hermes start-up); first-assistant seconds per task (session start to the first assistant message); timeouts. Row level: passed/30, passed per difficulty (A/B/C) and per category, wall total and median, first-turn wait = first-assistant seconds of the row's first task on a fresh server (cold prompt cache; the median over tasks is the warm figure), decode tok/s p50 (server log, requests of 32+ tokens), measured peak VRAM over the whole row (nvidia-smi every 2 s) and the VRAM after load.

## tiers, picks, ties (pre-registered)

- tier of a row = the smallest card in {8, 12, 16, 24, 32} GB with `peak_mib + 768 <= card * 1024` (the card_fit.py convention; 768 MiB driver headroom; card GB label = GiB). A row that needs expert offload is labelled offload and also lists the file size (system RAM need).
- pick for card C = the highest pass count among the rows whose tier is C or smaller. A lead of fewer than 3 tasks (of 30) over the next row is called a TIE and no pick is made from one run. A paired exact (two-sided) McNemar on per-task outcomes between the top two is reported next to it and is never decisive on one run.
- spread test: if the 4B row (Spark) scores 27 or more of 30, the set does not spread and the readout says so; if the best row scores 5 or fewer, the set is a floor. No task is re-frozen or dropped after outputs exist.
- a row that does not load (out of memory at every offload step tried) is a result: NOFIT with the number in the server log, reported as such.
- single run per row. Repeats go only to the top two rows of a card whose lead is under 3 tasks or whose pick will be quoted; that is a later item, not this one.

## honest limits (written before any number)

30 own tasks, one sampled run per row at the maker's sampling (not greedy), not a public benchmark; single-run differences of 1-2 tasks are noise; the 5090 is a stand-in for the smaller cards; the agent works as the box user with default approvals (blocks are counted, not hidden); thinking-on defaults differ by family (the reasoning-effort rule above); one install of Hermes (a newer Hermes may behave differently); q8_0 KV accuracy on these tasks is untested.
