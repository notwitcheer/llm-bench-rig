# chess-duel (queue item 81)

Two local models each get one Hermes Agent `/goal` hour to write a UCI chess engine in a single Python file from the same prompt. A pre-registered gate the agent never sees checks each engine, then they play a 20-game match, then the clip is rendered. Staged 2026-11-03, not yet run.

Two trees, on purpose (the agent has a terminal and can read everything under `~/voxelbench`):

| repo path | capsule path | who can see it |
|---|---|---|
| `prompts/`, `scripts/` | `~/voxelbench/chess-duel/` | the agent: prompt, driver, in-run smoke gate and its frozen data (no move generator, no criterion) |
| `final-gate/`, `render/`, `stubs/`, `tests/` | `~/.vxfinal/chess/` | nobody during a run: referee rules, final gate, frozen perft numbers, baseline, match runner, CPU phase, renderers, stub engines, mock model |

`scripts/deploy_capsule.sh` copies the two sets. The item's PLAN greps every agent-visible path (and every queued item) for the final gate's file names and numbers and refuses to start on a hit.

- `final-gate/chesslib.py` the referee's own chess rules; `tests/test_chesslib.py` checks them against the frozen node counts from chessprogramming.org/Perft_Results (fetched 2026-11-03, `final-gate/perft_ref.json`) and, with `--pychess`, against python-chess as a test oracle only (60 random games, 9,297 plies, 0 disagreements).
- `stubs/build_stubs.py` builds a known-good engine and eight deliberately broken ones; `tests/test_stubs.py` asserts the gate's verdict on each (all as pre-registered).
- `tests/e2e_mock.sh` runs a whole agent leg on a mock model: isolated `HERMES_HOME`, `/goal`, the in-run gate, then the final gate.
- `scripts/chess-smoke.sh` is the short real-model smoke (not part of the item).
- The pre-registration (gate, line-1 variants, match result rule, Elo anchor, confounds) is the header of `../queue/81-chess-duel.sh`.
