#!/bin/bash
# rc_smoke.sh: GPU smoke of the replay-cache item (staging proof, never counted; slug -smoke). 5 real source solves on the real selection (3 held-out + 2 near-miss)
# + 1 real builder loop through to a validated entry (template 320, instance 0 of main) + the sizing report. Wrapped by the caller in
# `flock -n /home/witcheer/.gpu-smoke.lock`. Donald is drained and restored ONLY through night-lib (nl_donald_stop / nl_donald_restore). Hard cap: SMOKE_MIN minutes.
set -uo pipefail
MANIFEST=/tmp/rc-smoke-manifest.json; export MANIFEST
NL_TAG=rcsmoke
. /home/witcheer/night-lib.sh
RC=/home/witcheer/esoteric/replay-cache; CODE=$RC/code
SRV=/home/witcheer/llama.cpp-donald/build/bin/llama-server
MODEL=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
. $CODE/regime.env
RUN=$RC/run/smoke; SEL=$RC/run/sel-real.json
SMOKE_MIN=${SMOKE_MIN:-14}
export RC_DATA=$RC/data
mkdir -p "$RUN"
T0=$(date +%s); DEADLINE=$(( T0 + SMOKE_MIN*60 ))
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
SP=""; DSTOP=0
cleanup() {
  [ -n "$SP" ] && { kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done; kill -KILL "$SP" 2>/dev/null; sleep 3; }
  if [ "$DSTOP" = 1 ]; then
    for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && break; sleep 2; done
    nl_donald_restore
    for i in $(seq 1 60); do H=$(curl -s -m 5 localhost:8090/health); [[ "${H,,}" == *ok* ]] && { nl_say "Donald healthy again after ~$((i*5)) s"; return 0; }; sleep 5; done
    nl_say "WARNING: Donald /health did not answer within 5 min"
  fi
}
trap cleanup EXIT
nl_say "rule 3 (live): compute apps: $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';') | tmux: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ') | active: $(cat ~/idleq-active.json 2>/dev/null | cut -c1-120)"
case " $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ') " in *" idleq "*) nl_say "ABORT: idleq is live"; exit 2;; esac
GP=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | grep -v llama-server | head -3)
[ -z "$GP" ] || { nl_say "ABORT: a foreign GPU process is live: $GP"; exit 2; }
nl_say "draining Donald (night-lib nl_donald_stop)"; DSTOP=1; nl_donald_stop
for i in $(seq 1 60); do v=$(mem_used); [ "${v:-99999}" -lt 1024 ] && break; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || { nl_say "ABORT: card still ${v} MiB after the drain"; exit 2; }
T_CARD=$(date +%s)
"$SRV" --model "$MODEL" --host 127.0.0.1 --port $GPORT $SRV_FLAGS --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" > "$RUN/server.log" 2>&1 &
SP=$!
for i in $(seq 1 120); do H=$(curl -s -m 3 http://127.0.0.1:$GPORT/health || true); [[ ${H,,} == *ok* ]] && break; kill -0 "$SP" 2>/dev/null || break; sleep 2; done
[[ ${H,,} == *ok* ]] || { nl_say "FAIL: server did not come up"; exit 1; }
nl_say "server up after $(( $(date +%s) - T_CARD )) s, VRAM $(mem_used) MiB"
( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=power.draw,memory.used,temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> "$RUN/gpu.csv"; sleep 2; done ) >/dev/null 2>&1 &
GS=$!
nl_say "source: ${RC_SMOKE_ITEMS:-5} real solves (half held-out, half near-miss)"
python3 "$CODE/rc_gpu.py" source --base http://127.0.0.1:$GPORT --out "$RUN" --sel "$SEL" --deadline $DEADLINE --slots $NP --max-tokens $MAXTOK --limit-items ${RC_SMOKE_ITEMS:-5} --min-tail-min 1 2>&1 | tail -8
nl_say "builder: ${RC_SMOKE_TPL:-1} real loop(s)"
python3 "$CODE/rc_gpu.py" build --base http://127.0.0.1:$GPORT --out "$RUN" --sel "$SEL" --deadline $DEADLINE --slots $NP --max-tokens $MAXTOK --limit-templates ${RC_SMOKE_TPL:-1} --min-tail-min 1 2>&1 | tail -10
curl -s -m 5 http://127.0.0.1:$GPORT/metrics | grep -E "^llamacpp:(tokens_predicted_total|tokens_predicted_seconds_total|prompt_tokens_total)" | tr '\n' ' ' > "$RUN/metrics-end.txt"
kill $GS 2>/dev/null
nl_say "card time of the smoke: $(( ($(date +%s) - T_CARD) / 60 )) min $(( ($(date +%s) - T_CARD) % 60 )) s ; VRAM peak $(cut -d, -f3 "$RUN/gpu.csv" | sort -n | tail -1) MiB ; power max $(cut -d, -f2 "$RUN/gpu.csv" | sort -n | tail -1) W"
python3 "$CODE/rc_smoke_report.py" "$RUN" > "$RUN/smoke-report.stdout" 2>&1; tail -30 "$RUN/smoke-report.stdout"
nl_say "smoke done"
