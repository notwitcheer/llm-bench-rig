# desktop-arena (voxelbench/desktop-arena): which local model can drive a Linux desktop in Hermes Agent

STAGED harness for GPU queue item 82 (not committed here: the gated push waits for the operator's yes). Everything is CPU-provable; the card is only needed for the model server.

- `desk_mkhome.py`  one entrant's private HERMES_HOME (not a profile of the main install): terminal.backend docker, desktop sandbox image, toolset computer_use only,
  tools.tool_search off, approvals STANDARD + command_allowlist grants `cua:<action>:<background|foreground>` (the same pre-granted actions for every entrant).
- `desk_tasks.py`   the 8 tasks (seed + prompt + scorer + reference solution). `python3 desk_tasks.py --prove <home>` runs the controls (fresh seed fails, reference passes).
- `desk_runner.py`  per task: fresh desktop, seed, frame grab of the agent's display, `hermes -t computer_use chat -q`, hard wall cap, score from file state.
- `desk_clip.py`    three separate captures -> one split-screen clip + scoreboard (PIL + ffmpeg pipe).
- `desk_mock.py`    scripted OpenAI-compatible model (tool-call script, `@el:<regex>` element resolution) for the CPU proof.
- `desk_lib.py`     docker exec helpers (harness side only).
- `image/Dockerfile` official `nousresearch/hermes-sandbox:desktop` + mousepad thunar gnumeric.
- `82-desktop-arena.sh` the queue item (header, PLAN/RUN, window rule, Donald drain + EXIT-trap restore). `ARENA_MOCK=<script.json> TASKS=t1` runs the whole orchestration with the mock and no card.
