#!/bin/bash
# e2e_mock.sh <mode> [slug-suffix]: the whole agent leg on a MOCK model (CPU only, no card): isolated HERMES_HOME + venv hermes_cli + /goal + in-run quick gate + pane recorder,
# then the final gate on what the "agent" wrote. mode = good | bad_import | none   (the mock's FIRST write; after a gate failure the mock writes the good engine)
# usage (capsule): bash ~/.vxfinal/chess/tests/e2e_mock.sh good     artefacts in ~/voxel-stage/chess-e2e/runs/<slug>/
set -uo pipefail
MODE=${1:-good}; SUF=${2:-$MODE}
FX=/home/witcheer/.vxfinal/chess
V=/home/witcheer/voxel-stage/chess-e2e          # staging copy of the runs dir (never ~/voxelbench/runs)
CD=/home/witcheer/voxelbench/chess-duel
SLUG=e2e-$SUF; AGENT=m$(echo "$SUF" | tr -cd a-z0-9 | head -c 8); PORT=${PORT:-8583}; ARENA=${ARENA:-/tmp/chess81}
mkdir -p "$V/runs/$SLUG" "$ARENA"
python3 $FX/stubs/build_stubs.py $V/stubs > /dev/null || exit 2
ss -ltnH "sport = :$PORT" | grep -q . && { echo "[e2e] port $PORT is taken by someone else: pick another PORT"; exit 2; }
echo "[e2e] $(date +%T) mock model on :$PORT (first write: $MODE)"
tmux new-session -d -s chess81-mock-$SUF "MOCK_DIR=$V/stubs MOCK_FIRST=$MODE MOCK_RATE=${MOCK_RATE:-120} python3 $FX/tests/chess_mock.py $PORT > $V/runs/$SLUG/mock.log 2>&1"
for i in $(seq 1 20); do curl -s 127.0.0.1:$PORT/health | grep -qi ok && break; sleep 0.5; done
curl -s 127.0.0.1:$PORT/health | grep -qi ok || { echo "[e2e] mock did not start"; exit 2; }
{ cat $CD/prompts/chess-v1.txt; printf '\nThe working directory is %s. The final file must be exactly %s/engine.py (copy this path, never retype it).\n' "$ARENA/$AGENT" "$ARENA/$AGENT"; } > $V/runs/$SLUG/prompt.txt
QG="python3 $CD/scripts/quick_gate_chess.py $ARENA/$AGENT/engine.py $V/runs/$SLUG/qgate"
t0=$(date +%s)
PORT=$PORT ARENA=$ARENA V=$V VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="$QG" bash $CD/scripts/chess_agent.sh $SLUG $AGENT mock-chess 131072 ${WALL:-6} ${TURNS:-8} $V/runs/$SLUG/prompt.txt > $V/runs/$SLUG/agent.out 2>&1
echo "[e2e] $(date +%T) agent leg rc=$? after $(( $(date +%s) - t0 )) s: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
tmux kill-session -t "=chess81-mock-$SUF" 2>/dev/null
ENG=$ARENA/$AGENT/engine.py
if [ -f "$ENG" ]; then
  GATE_PERFT_TIMEOUTS=60,60,60,300,600 GATE_SKIP_EXTRAS=${GATE_SKIP_EXTRAS:-1} python3 $FX/chess_final_gate.py "$ENG" $V/runs/$SLUG/gate > $V/runs/$SLUG/gate.log 2>&1; echo "[e2e] final gate rc=$?: $(tail -1 $V/runs/$SLUG/gate.log)"
else
  mkdir -p $V/runs/$SLUG/gate; echo '{"pass": false, "no_engine": true}' > $V/runs/$SLUG/gate/gate.json; echo "[e2e] no engine.py written: gate.json says no_engine"
fi
echo "[e2e] quick gate result: $(cat $V/runs/$SLUG/qgate/gate.json 2>/dev/null | tr -d '\n ' | head -c 300)"
echo "[e2e] pane frames: $(wc -l < $V/runs/$SLUG/pane-rec/frames.jsonl 2>/dev/null)"
