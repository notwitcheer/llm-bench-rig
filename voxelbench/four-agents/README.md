# four-agents: four Hermes Agent sessions, one GPU, one task, filmed

queue item `queue/80-four-agents-one-gpu.sh` (staged 2026-11-03, enqueue is the operator's call). capsule path `~/voxelbench/four-agents/`.

one llama-server (`--parallel 4`, pooled KV) serves four Hermes Agent sessions at once. each agent has its own `HERMES_HOME` (not a profile of the main install, so the harness stays isolated from the always-on gateway), its own port on a counting proxy, and the same prompt (`prompts/lighthouse-v1.txt`, one animated isometric voxel lighthouse island as a single canvas page). four tmux/pty panes start on the same Enter; a HUD logs per-agent and aggregate tok/s; a 2x2 clip with a result mosaic is rendered on capsule.

| file | what it does |
|---|---|
| `four_mkhome.py` | one agent's private `HERMES_HOME` (config, SOUL, `installs/` + `tools/` symlinked from the main root) |
| `four_proxy.py` | counting proxy, one port per agent, 1 Hz HUD sampler (`/metrics`, `/slots`, nvidia-smi) -> `hud.jsonl`, `requests.jsonl` |
| `four_record.py` | pty recorder (pyte), types the prompt, barrier, Enter at GO, done / wall detection; JSONL events on one epoch clock |
| `four_race.sh` | glue: proxy, four homes, four tmux sessions `f80-a..d`, barrier, wait, copy work dirs + state.db, stop only what it started |
| `four_gate.py` | CPU liveness gate (no console/page errors, no network, canvas, not blank, animated) + 12 s capture |
| `four_probe.py` | solo vs four-at-once tok/s on the same server (the HUD baseline) |
| `four_render.py` | 1920x1080 clip: fixed 2x2 grid, per-pane tok/s + clock, aggregate tok/s + sparkline, timelapse middle, leak mask, result card |
| `four_results.py` | 2x2 mosaic of the four pages + concat onto the race clip |
| `four_summary.py` | `summary.json`: fit rows, probe, race, gates, request stats, headline |
| `four_mock.py` | llama-server-shaped mock for the CPU PASS path (test aid only) |

CPU proof (no model): `four_mock.py 8196`, then `FOUR_STUB_UPSTREAM=http://127.0.0.1:8196 FOUR_SMOKE=1 FOUR_WALL_MIN=5 bash ~/voxel-stage/80-four-agents-one-gpu.sh`.
traps and numbers: vault `99-personal/benchmark-rig/four-agents-80-status.md`, rows L42-L45 in `../HARDENING.md`.
