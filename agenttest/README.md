# agenttest: 10 short Hermes Agent tasks, one fresh session each

Durable home of the agent task test first run on Muse Glimmer 30B (2026-10-20, draft
`../drafts/2026-10-21-muse-glimmer-agent.md`). The original harness lived in
`/opt/data/cache/scratch/museagent/` and was pruned by the 24 h scratch cleaner, so this folder
rebuilds it (2026-10-22). Never keep a harness only in scratch again.

## what is original and what is rebuilt

- **prompts: original.** `tasks.json` holds the first user message of each Muse session, copied
  verbatim from `/opt/data/tmp/museagent/home/state.db` (sessions 20261020_1309..1314). Only the
  work path changed: `/opt/data/tmp/museagent/work` -> `/opt/data/tmp/sparkagent/work`.
  `muse_prompt_sha256` hashes the original text. The t01 line (`muse glimmer was here`) is kept as is.
- **seeds: recovered, not regenerated.** The Muse input files survived in
  `/opt/data/tmp/museagent/work` (mtime 2026-10-20 13:09:29 UTC, never touched by the model). They
  are frozen in `seed_data/` byte for byte (sha256 in `seed_data/PROVENANCE.json`). Two seed files
  the model edited (t04 `stats.py`, t08 `config.json`) were reconstructed from the Muse transcript's
  `read_file` output; both sizes match the logged `file_size` (85 B, 200 B). `t09/run_report.sh`
  carries the work path, stored with a `{WORK}` placeholder. `gen_seeds.py` is deterministic (no
  randomness): it wipes the work root and writes these files.
- **checkers: re-implemented.** `checkers/check_tNN.py`, expected values derived from `seed_data/`
  (never from files the agent can edit). Cross-check: the rebuilt checkers pass all 10 surviving
  Muse outputs, and the rebuilt metric extractor reproduces the Muse draft's 46 tool calls / 4 tool
  errors / 0 malformed exactly (`results/muse-crosscheck-2026-10-22.json`).

So a new run shares prompts and seed content with the Muse 10/10; the checkers are new code that
agrees with the old verdicts on the Muse outputs.

## files

- `tasks.json` frozen prompts + smoke prompt + work root
- `seed_data/` frozen inputs, `gen_seeds.py` writes a fresh tree
- `checkers/` one checker per task, `check(work) -> (pass, reason)`; CLI `python3 check_t04.py <work>`
- `hand_solutions.py` hand solution + one plausible near miss per task (proof only)
- `prove_checkers.py` per task: fresh seed must FAIL, near miss must FAIL, hand solution must PASS;
  writes `results/checker-proof-<date>.json`. Run before any model output.
- `run_agent.py` runner (mercury): fresh seeds once, then per task
  `HERMES_HOME=<isolated home> hermes chat -q <prompt> --format stream-json --source tool --run-budget 900`
  from the work root, stdin closed (non-TTY one-shot, default approvals), transcripts `tNN.jsonl`,
  checker verdict + state.db metrics `tNN.result.json`, totals `results.json`.
- `parse_server_log.py` llama-server per-request timings (decode p50/p90) + gpu sampler summary
- serve script: `../scripts/sparkagent-serve.sh` (capsule `~/sparkagent-serve.sh`)

## metric definitions

- tool calls: entries in assistant `tool_calls` (state.db messages)
- tool error: a tool result with `status: error`, `success: false`, a non-null `error`, or a non-zero
  `exit_code` (includes Hermes' headless approval blocks and designed failures)
- malformed call: Hermes' unknown-tool or invalid-JSON-arguments reply, or a raw tool-call string left
  in a final assistant message
- wall s: subprocess start to exit, per task (includes Hermes start-up)

## isolated home

`/opt/data/tmp/<run>/home/config.yaml`: provider custom, `base_url http://127.0.0.1:18197/v1` (ssh -L
to capsule 127.0.0.1:8197), `context_length 65536`, `terminal.cwd` = work root. Nothing else: approvals
stay at the default (`smart`, `single_query_mode: deny`). Never the default profile.
