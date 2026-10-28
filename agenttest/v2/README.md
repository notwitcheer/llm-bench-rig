# agenttest v2: 30 offline Hermes Agent tasks, one fresh session per task

Built 2026-10-28 for lane L1 (Hermes Agent on 8 / 12 / 16 / 24 / 32 GB: the measured pick per VRAM tier, C1 = models already on disk). v1 (`../`, 10 short tasks) hit a ceiling (a 4B scored 10/10); v2 is multi-step. Read `PREREG.md` first: design and scoring were fixed before any model output.

## layout

- `tasks/t_*.py` the 30 tasks. Each class: `prompt` (with `{T}` = the task folder), `seed(T)`, `check(T) -> (pass, reason, frac)`, `solve(T)` (hand solution, proof only), `near(T)` (a plausible wrong answer, proof only).
- `common.py` registry + helpers (deterministic git fixtures, tree hash). `freeze.py` writes `tasks.json` (prompts, budgets, seed hashes) and `FROZEN.sha256`; `freeze.py --check` verifies.
- `prove_checkers.py` per task: empty folder fails, fresh seed fails, near miss fails, hand solution passes with partial score 1.0, seed deterministic and path independent. Output `results/checker-proof-<date>.json`. Run before any run.
- `rows.json` the C1 rows (run order, file, bytes, arch, maker sampling + source, provisional tier). `row_info.py` prints them for the queue item.
- `run_agent.py` the row runner (capsule side): profile per row, seed + `hermes -p <profile> chat -q ... --format stream-json --source tool` per task, checker, state.db metrics, resumable per task, `summary.json` per row.
- `fit_probe.py` VRAM at depth (one ~14k-token request) for the q8_0 KV fit leg. `parse_server_log.py` decode tok/s from the llama-server log.
- `aggregate.py` tier table + readout under the pre-registered rules; `chart_l1_tiers.py` the per-tier card (`/opt/data/venvs/charts/bin/python`).
- queue item: `../../scripts/capsule-queue/73-agenttest-v2-c1.sh` (night-lib contract: DRYRUN, preflight, Donald drain + EXIT-trap restore, hard stop, resumable per row, manifest json).

## run it

On capsule: code in `~/agenttest-v2/code`, results in `~/agenttest-v2/results/c1/<row>/`, manifest `~/agenttest-v2-c1-status.json`. The runner runs ON capsule (not through a tunnel): a weekend-long unattended run must not depend on an ssh -L from another box, and the hard stop, Donald drain/restore and resume logic live next to the card. The cost: the agent runs as the box user with Hermes' default approvals (blocks are counted, not hidden), in a freshly seeded folder per task.

```
python3 prove_checkers.py            # ALL_OK 30/30 or stop
python3 freeze.py --check            # tasks.json = registry + seeds
DRYRUN=1 bash 73-agenttest-v2-c1.sh  # preflight + plan, no card
```

## metric definitions

pass = the checker is true. tool error = a tool result with an error status, non-zero exit code or error text that is not an approval block. approval block = Hermes' headless BLOCKED verdict. malformed = unknown tool name, invalid arguments, or a raw tool-call string in the final text. first-assistant seconds = session start to the first assistant message. wall = process start to exit (includes Hermes start-up). Tier and pick rules: `PREREG.md`.
