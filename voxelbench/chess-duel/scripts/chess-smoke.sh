#!/bin/bash
# chess-smoke.sh: GPU SMOKE of the real-model path of item 81 (<20 min of card, only before 14:00 or after 17:15 Paris on 2026-11-03, never while a sibling owns the card).
# One Qwen3.8-27B Q6_K agent leg with the item's exact server flags and harness, a SHORT wall (default 10 min), on port 8581; then the final gate on whatever engine.py it wrote.
# NOT part of the queue item. refuses (never kills anything) when a compute app other than Donald holds the card, when idleq is live, or when port 8581 is taken.
# usage (capsule, detached tmux): bash ~/voxel-stage/chess-smoke.sh     env: WALL (10)
set -uo pipefail
WALL=${WALL:-10}
V=/home/witcheer/voxel-stage/chess-smoke; CD=/home/witcheer/voxelbench/chess-duel; FX=/home/witcheer/.vxfinal/chess
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
MF=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; ALIAS=Qwen3.8-27B-Q6_K.gguf
PORT=8581; ARENA=/tmp/chess81; SLUG=chess81-smoke; AGENT=smoke
say() { echo "[chess81-smoke] $(date +%H:%M:%S) $*"; }
say "ownership check (rule 3): compute apps, tmux, idleq-active"
DPID=$(systemctl show -p MainPID --value llama-server.service 2>/dev/null)
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
say "compute apps: ${APPS:-none} | Donald MainPID ${DPID:-?}"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  q=$p; mine=0; while [ -n "$q" ] && [ "$q" != 1 ] && [ "$q" != 0 ]; do [ "$q" = "$DPID" ] && { mine=1; break; }; q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' '); done
  [ "$mine" = 1 ] || { say "REFUSE: GPU pid $p is not Donald's tree"; exit 4; }
done
tmux has-session -t =idleq 2>/dev/null && { say "REFUSE: idleq is live"; exit 4; }
say "tmux: $(tmux ls -F '#{session_name}' | tr '\n' ' ')"; say "idleq-active: $(cat /home/witcheer/idleq-active.json 2>/dev/null | cut -c1-200)"
ss -ltnH "sport = :$PORT" | grep -q . && { say "REFUSE: port $PORT taken"; exit 4; }
HR=$(date +%H); [ "$HR" -lt 14 ] || [ "$HR" -ge 18 ] || { say "REFUSE: GPU is queue item 79's from 14:10 to 17:10"; exit 4; }
SRV=""; DRAINED=0
restore() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; SRV=""; tmux kill-session -t "=chess81-$AGENT" 2>/dev/null; sleep 3
  [ "$DRAINED" = 1 ] && { say "restoring Donald"; sudo -n systemctl start llama-server.service; for i in $(seq 1 150); do curl -s -m 3 127.0.0.1:8090/health | grep -qi ok && { say "Donald healthy after ~$((i*2)) s"; break; }; sleep 2; done; DRAINED=0; }; }
trap restore EXIT
T0=$(date +%s)
say "draining Donald"; DRAINED=1; sudo -n systemctl stop llama-server.service; sleep 5
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || { say "drain never cleared"; exit 1; }
mkdir -p $V/runs/$SLUG
$BIN --model $MF --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size 131072 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --alias $ALIAS --metrics \
  --reasoning-budget 16384 --reasoning-budget-message "Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns." \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --spec-type draft-mtp --spec-draft-n-max 2 > $V/server.log 2>&1 &
SRV=$!
UP=0; for i in $(seq 1 150); do curl -s 127.0.0.1:$PORT/health | grep -qi ok && { UP=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
[ "$UP" = 1 ] || { say "server did not come up: $(grep -m1 -iE 'error|failed|out of memory' $V/server.log | head -c 200)"; exit 1; }
say "server up on :$PORT, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader), load took $(( $(date +%s) - T0 )) s"
curl -s 127.0.0.1:$PORT/props > $V/runs/$SLUG/props.json
say "sampling defaults: $(python3 -c "import json;g=json.load(open('$V/runs/$SLUG/props.json')).get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p')})" 2>/dev/null)"
{ cat $CD/prompts/chess-v1.txt; printf '\nThe working directory is %s. The final file must be exactly %s/engine.py (copy this path, never retype it).\n' "$ARENA/$AGENT" "$ARENA/$AGENT"; } > $V/runs/$SLUG/prompt.txt
export DEADLINE_EPOCH=$(( T0 + 17*60 ))
PORT=$PORT ARENA=$ARENA V=$V VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="python3 $CD/scripts/quick_gate_chess.py $ARENA/$AGENT/engine.py $V/runs/$SLUG/qgate" \
  bash $CD/scripts/chess_agent.sh $SLUG $AGENT $ALIAS 131072 $WALL 20 $V/runs/$SLUG/prompt.txt > $V/runs/$SLUG/agent.out 2>&1
say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
curl -s 127.0.0.1:$PORT/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total)" > $V/runs/$SLUG/server-metrics.txt
kill -9 $SRV 2>/dev/null; SRV=""; sleep 3
restore
say "card released after $(( ($(date +%s) - T0) / 60 )) min"
python3 /home/witcheer/voxelbench/scripts/metrics.py "$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['home'])")" $V/runs/$SLUG/metrics.json > /dev/null 2>&1
ENG=$V/runs/$SLUG/work/engine.py
if [ -f $ENG ]; then say "engine.py: $(wc -l < $ENG) lines, $(wc -c < $ENG) bytes; final gate (CPU)"; python3 $FX/chess_final_gate.py $ENG $V/runs/$SLUG/gate > $V/runs/$SLUG/gate.log 2>&1; say "final gate rc=$?: $(tail -1 $V/runs/$SLUG/gate.log)"; else say "no engine.py written in $WALL min"; fi
say "quick gate: $(cat $V/runs/$SLUG/qgate/gate.json 2>/dev/null | tr -d '\n' | head -c 400)"
say "DONE"
