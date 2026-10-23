#!/bin/bash
# Spark-X2.6-4B Q4_K_M served for the Hermes Agent 10-task test (2026-10-22, mercury). adapted from
# ~/museagent-serve.sh. SOLO booking: waits (60 s polls) until the card is free of every job except
# possibly Donald, gives up at 13:15 CEST, unloads Donald via llama-swap (never stops the service),
# serves on 127.0.0.1:8197 until ~/sparkagent-stop or the 13:45 CEST hard stop.
# ~/sparkagent-BUSY exists while we hold the card (removed on ANY exit); ~/sparkagent-DONE on exit.
set -u
export TZ=Europe/Paris
BIN=$HOME/llama.cpp-bolt/build/bin
M=$HOME/models/spark-x26-4b/Spark-X2.6-4B-Q4_K_M.gguf
REPO=XHToken/Spark-X2.6-4B-GGUF
FN=Spark-X2.6-4B-Q4_K_M.gguf
GIVEUP=$(date -d '2026-10-22 13:15:00' +%s)
HARD_STOP=$(date -d '2026-10-22 13:45:00' +%s)
GPULOG=$HOME/sparkagent-gpu.log
SRVLOG=$HOME/sparkagent-server.log
say(){ echo "[$(date +%H:%M:%S) CEST] $*"; }
SP=""; HELD=0; STATUS="exited before load"
cleanup(){
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 4; [ -n "$SP" ] && kill -9 $SP 2>/dev/null
  [ -n "${SAMP:-}" ] && kill $SAMP 2>/dev/null
  rm -f $HOME/sparkagent-READY
  if [ "$HELD" = 1 ]; then
    for i in $(seq 1 20); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$U" -lt 1000 ] && break; sleep 1; done
    say "server stopped; vram_end ${U:-?} MiB"
  fi
  rm -f $HOME/sparkagent-BUSY
  echo "sparkagent finished $(date -Is): $STATUS" > $HOME/sparkagent-DONE
  say "BUSY removed, DONE written ($STATUS)"
  say "donald :8090 health: $(curl -s -m5 http://127.0.0.1:8090/health) llama-server.service: $(systemctl is-active llama-server.service)"
  say SERVE_END
}
trap cleanup EXIT
trap 'STATUS="killed by signal"; exit 1' INT TERM HUP

# ---- preflight (no card) ----
$BIN/llama-server --version 2>&1 | grep -i version
NLIB=$(strings $BIN/../bin/libllama.so 2>/dev/null | grep -ci spark2_5); say "libllama spark2_5 hits: $NLIB"
[ "$NLIB" -gt 0 ] || { STATUS="build lacks spark2_5"; exit 2; }
WANT=$(curl -s -m 30 "https://huggingface.co/api/models/$REPO/tree/main" | python3 -c "import json,sys; d=json.load(sys.stdin,strict=False); print([f['size'] for f in d if f['path']=='$FN'][0])")
HAVE=$(stat -c %s "$M")
say "size check: local $HAVE vs HF tree API $WANT"
[ -n "$WANT" ] && [ "$HAVE" = "$WANT" ] || { STATUS="size mismatch ($HAVE vs ${WANT:-none})"; exit 2; }

# ---- wait for a solo card ----
busy_reason(){
  local r="" pid ppid pcomm
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' '); pcomm=$(ps -o comm= -p "${ppid:-1}" 2>/dev/null)
    [ "$pcomm" = "llama-swap" ] && continue          # Donald (llama-swap child) is allowed
    r="$r gpu:$pid:$(ps -o comm= -p $pid 2>/dev/null)"
  done
  local t; t=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^(vx-|voxel-|prz|dgz|hkv4)' | tr '\n' ',')
  [ -n "$t" ] && r="$r tmux:$t"
  if [ -f $HOME/idleq-active.json ]; then
    local s; s=$(python3 -c "import json; print(json.load(open('$HOME/idleq-active.json')).get('script',''))" 2>/dev/null)
    if tmux has-session -t =idleq 2>/dev/null || { [ -n "$s" ] && ps -eo args | grep -F "$s" | grep -vq grep; }; then r="$r idleq:$(basename "$s")"; fi
  fi
  echo "$r"
}
CLEAR=0
while :; do
  NOW=$(date +%s)
  [ "$NOW" -lt "$GIVEUP" ] || { STATUS="gave up at 13:15 CEST, card never free"; say "$STATUS"; exit 3; }
  R=$(busy_reason)
  if [ -z "$R" ]; then CLEAR=$((CLEAR+1)); say "card clear ($CLEAR/2)"; else CLEAR=0; say "busy:$R"; fi
  if [ "$CLEAR" -ge 2 ]; then
    echo "sparkagent (mercury subagent) holds the card since $(date -Is): Spark-X2.6-4B Hermes Agent 10-task test, ~20 min, hard stop 13:45 CEST" > $HOME/sparkagent-BUSY
    HELD=1
    say "BUSY written; unloading Donald: $(curl -s -m10 -X POST http://127.0.0.1:8090/api/models/unload)"
    for i in $(seq 1 60); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$U" -lt 1000 ] && break; sleep 1; done
    R=$(busy_reason)
    if [ "$U" -lt 1000 ] && [ -z "$R" ]; then say "vram before load $U MiB, card solo"; break; fi
    say "not solo after unload (vram $U MiB,$R); back to waiting"; rm -f $HOME/sparkagent-BUSY; HELD=0; CLEAR=0
  fi
  sleep 60
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/apps_before /'

# ---- serve ----
$BIN/llama-server -m $M -ngl 99 -fa on -c 65536 --jinja --port 8197 --host 127.0.0.1 > $SRVLOG 2>&1 & SP=$!
echo $SP > $HOME/sparkagent-server.pid
for i in $(seq 1 90); do curl -s -m2 http://127.0.0.1:8197/health | grep -qi ok && break; sleep 2; done
say "health: $(curl -s -m2 http://127.0.0.1:8197/health)"
curl -s -m5 http://127.0.0.1:8197/props > $HOME/sparkagent-props.json
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/vram_loaded /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/apps_loaded /'
touch $HOME/sparkagent-READY
STATUS="served"
( while :; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr -d ' ' | tr '\n' ';') tmux=$(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ',')"
    sleep 5
  done ) >> $GPULOG 2>&1 & SAMP=$!
while [ "$(date +%s)" -lt "$HARD_STOP" ]; do
  kill -0 $SP 2>/dev/null || { STATUS="server died"; say "$STATUS"; exit 1; }
  [ -f $HOME/sparkagent-stop ] && { rm -f $HOME/sparkagent-stop; STATUS="served, stopped by stop file"; say "stop file seen"; exit 0; }
  sleep 5
done
STATUS="served, HARD STOP 13:45 reached"; say "$STATUS"
