# wedit95: a local Qwen3.8-27B as the weight editor of Qwen3.5-0.8B (ran 2026-11-07)

Item `95-weight-editor.sh` for the capsule idle queue. Question: can a local Qwen3.8-27B (Q6_K, llama.cpp) driving a Hermes Agent session find a weight edit that cures Qwen3.5-0.8B's thinking loops, and does it match a scripted pipeline? Four arms on 40 held-out GSM8K test problems: base, the published MIT CSAIL edit (bit-exact), a scripted pipeline, the editor's best-by-dev edit.

- `95-weight-editor.sh`: the queue item (night-lib contract, DRYRUN-safe).
- `code/PREREG.md` + `code/FROZEN.sha256`: pre-registration and the hashes the item checks before it runs (paths in the hash list are relative to the staging layout, `../ref/materials/` holds the published edit files, not copied here).
- `code/wd_summary.py --selftest`: verdict rule, 11 branches.
- `code/wd_service.py`, `wd_cli.py`: the sandbox tool the editor sees (`wd`).
- `code/render_wedit.py`: the clip (CPU only).

Mock-render and mock-take scaffolding (`wd_mock.py`, `wd_make_mock.py`) never produce counted output. Hosted-model claims are out of scope.

## result (run Sat 2026-11-07 02:56-04:31 CET, one RTX 5090)

verdict on the pre-registered rule (`code/wd_summary.py`): **EDITOR COULD NOT MATCH THE SCRIPT**.

40 held-out GSM8K problems, thinking on, 16,384-token cap, one try each:

| arm | thinking never finished (loops) | answered correctly |
|---|---|---|
| Qwen3.5-0.8B as released | 33 / 40 | 6 / 40 |
| the published MIT CSAIL edit, applied bit-exact | 16 / 40 | 17 / 40 |
| scripted pipeline (no model in the loop) | 16 / 40 | 16 / 40 |
| the editor's edit (Qwen3.8-27B Q6_K in Hermes Agent, one 33-min session) | 34 / 40 | 6 / 40 |

the editor scored two candidates on the 20-problem dev set: its first edit looped more than the unedited model (19/20 vs 16/20), so it reversed the edit and doubled it (17/20), still worse than no edit. one-sided Fisher vs the unedited model: published p = 9.4e-5, editor p = 0.73. numbers: `run/summary.json` (the candidate path is relative to the staging dir).
