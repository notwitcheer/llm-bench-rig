#!/bin/bash
# deploy_capsule.sh: copy the chess-duel harness from the public-clone staging tree to capsule (run on mercury).
#   agent-VISIBLE set  -> capsule ~/voxelbench/chess-duel/   (prompt, driver, in-run quick gate + its frozen smoke data, pane recorder)
#   HIDDEN set         -> capsule ~/.vxfinal/chess/          (referee rules, final gate, frozen perft numbers, baseline, stubs, mock model, tests, CPU phase, renderers)
# Nothing with a move generator or a criterion may sit in the visible set: the agent has a terminal and can read ~/voxelbench. never rm anything on capsule.
set -euo pipefail
SRC=${SRC:-/opt/data/llm-bench-rig/voxelbench/chess-duel}
R=${R:-capsule-w}
ssh "$R" 'mkdir -p ~/voxelbench/chess-duel/prompts ~/voxelbench/chess-duel/scripts ~/.vxfinal/chess/stubs ~/.vxfinal/chess/tests ~/.vxfinal/chess/render'
scp -q "$SRC/prompts/chess-v1.txt" "$R:voxelbench/chess-duel/prompts/"
scp -q "$SRC"/scripts/{chess_agent.sh,chess_mkhome.py,pane_rec.py,quick_gate_chess.py,qgate_data.json} "$R:voxelbench/chess-duel/scripts/"
scp -q "$SRC"/final-gate/* "$R:.vxfinal/chess/"
scp -q "$SRC"/stubs/* "$R:.vxfinal/chess/stubs/"
scp -q "$SRC"/tests/* "$R:.vxfinal/chess/tests/"
scp -q "$SRC"/render/* "$R:.vxfinal/chess/render/" 2>/dev/null || true
scp -q "$SRC"/scripts-hidden/* "$R:.vxfinal/chess/" 2>/dev/null || true
ssh "$R" 'chmod +x ~/voxelbench/chess-duel/scripts/chess_agent.sh; chmod 700 ~/.vxfinal/chess; ls -la ~/voxelbench/chess-duel/scripts ~/.vxfinal/chess | head -40'
