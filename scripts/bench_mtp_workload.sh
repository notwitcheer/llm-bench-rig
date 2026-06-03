#!/usr/bin/env bash
# bench_mtp_workload.sh — MTP speedup by WORKLOAD TYPE. base vs MTP generation tok/s per workload.
#
# CRITICAL: the live llama-server.service has Restart=always and a slow (26GB) shutdown. A naive
# `stop; sleep 3; launch` races the port -> our server fails to bind and requests silently hit the
# OLD server (both "base" and "MTP" then read identical). So we GATE on a clean slate (port free +
# VRAM drained) before each launch, and verify our own PID is alive + bound before measuring.
# Trap-guarded: restarts the live service on exit.
set -uo pipefail

HOST=127.0.0.1; PORT="${PORT:-8090}"
SERVER_BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen3.6-27B-Q6_K.gguf}"
MTP_MODEL="${MTP_MODEL:-$HOME/models/Qwen3.6-27B-MTP-Q6_K.gguf}"
SERVICE="${SERVICE:-llama-server}"
CTX="${CTX:-65536}"; NGL="${NGL:-99}"; NPREDICT="${NPREDICT:-256}"
OUT="${OUT:-/tmp/mtp_workload.json}"
RESULTS="$(mktemp)"; PAYLOAD="$(mktemp)"; LOGDIR="$(mktemp -d)"

restart_service(){ echo "[wl] restarting $SERVICE" >&2; sudo systemctl start "$SERVICE" || true; }
trap restart_service EXIT INT TERM

wait_clear(){  # port free AND VRAM drained (our pre-launch gate)
  for _ in $(seq 1 120); do
    if ! curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
      [ "${u:-999999}" -lt 4000 ] && return 0
    fi
    sleep 1
  done
  echo "[wl] slate never cleared" >&2; return 1
}
wait_for_health(){ for _ in $(seq 1 180); do curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && return 0; sleep 1; done; return 1; }

prompt_for(){ python3 -c '
import sys
w=sys.argv[1]
P={
 "prose":"Write a thorough, detailed explanation (at least 150 words) of the tradeoffs of running large language models locally on a single consumer GPU: memory capacity, bandwidth, quantisation, and context length.",
 "Q&A":"Explain in detail, step by step, what a KV cache is in a transformer, why it grows with context length, and how it affects both inference speed and memory use.",
 "JSON":"Output a JSON array of 30 user objects, each with fields id, name, email, score, active. Output only the JSON.\n[",
 "code":"Write a thread-safe LRU cache in Python with get and put methods, full docstrings, type hints, and a short usage example.\n```python\n",
 "repetitive":"Output 60 sequential log lines, each exactly in the form STATUS: OK seq=<n> with n incrementing from 1.\nSTATUS: OK seq=1\n",
}
print(P[w])' "$1"; }

mkpayload(){ python3 -c '
import sys,json
json.dump({"prompt":sys.argv[1],"n_predict":int(sys.argv[2]),"temperature":0,"cache_prompt":False},open(sys.argv[3],"w"))' "$1" "$NPREDICT" "$PAYLOAD"; }

WORKLOADS="prose Q&A JSON code repetitive"

measure_config(){ # $1=label $2=model $3..=flags
  local label="$1" model="$2"; shift 2
  local log="$LOGDIR/$label.log"
  wait_clear || return
  "$SERVER_BIN" --model "$model" --n-gpu-layers "$NGL" --ctx-size "$CTX" --flash-attn on --parallel 1 \
      --host "$HOST" --port "$PORT" "$@" > "$log" 2>&1 &
  local pid=$!
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null || grep -q "couldn't bind" "$log"; then
    echo "[wl] $label FAILED TO LAUNCH (port race?): $(grep -iE 'bind|error' "$log" | tail -1)" >&2; return
  fi
  wait_for_health || { echo "[wl] $label unhealthy" >&2; kill "$pid" 2>/dev/null||true; return; }
  # prove it's OUR server: PID must still be alive and own the port
  kill -0 "$pid" 2>/dev/null || { echo "[wl] $label died after health" >&2; return; }
  mkpayload "warm up please respond" 16; curl -s "http://$HOST:$PORT/completion" --data @"$PAYLOAD" >/dev/null
  local w resp parsed
  for w in $WORKLOADS; do
    mkpayload "$(prompt_for "$w")" "$NPREDICT"
    resp=$(curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' --data @"$PAYLOAD")
    parsed=$(printf '%s' "$resp" | python3 -c 'import sys,json
try:
    t=json.load(sys.stdin)["timings"]; print("%.1f %d"%(t["predicted_per_second"],t["predicted_n"]))
except Exception: print("ERR")')
    if [ "$parsed" = "ERR" ] || [ -z "$parsed" ]; then echo "[wl] $label $w FAILED: ${resp:0:120}" >&2
    else echo "$label $w $parsed" >> "$RESULTS"; echo "[wl] $label $w -> $parsed" >&2; fi
  done
  kill "$pid" 2>/dev/null||true; wait "$pid" 2>/dev/null||true
}

sudo systemctl stop "$SERVICE" || true
echo "[wl] baseline..." >&2; measure_config baseline "$BASE_MODEL"
echo "[wl] mtp..." >&2;      measure_config mtp "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2 --cache-type-k q8_0 --cache-type-v q8_0

python3 - "$RESULTS" "$OUT" <<'PY'
import sys,json
base={}; mtp={}
for line in open(sys.argv[1]):
    p=line.split(maxsplit=3)
    if len(p)<4: continue
    lab,w,tps,pn=p[0],p[1],float(p[2]),int(p[3])
    if tps>1000 or pn<64: continue
    (base if lab=="baseline" else mtp)[w]=tps
cats=[w for w in ["prose","Q&A","JSON","code","repetitive"] if w in base and w in mtp]
cats.sort(key=lambda w: mtp[w]/base[w])
data={
 "type":"bar",
 "title":"MTP speedup depends on the workload, not the context",
 "subtitle":"Qwen3.6-27B · RTX 5090 · generation tok/s · base vs MTP self-speculative",
 "y_label":"generation tok/s",
 "categories":cats,
 "series":[{"name":"base","values":[round(base[w],1) for w in cats],"color":"#e06060"},
           {"name":"MTP","values":[round(mtp[w],1) for w in cats],"color":"#e8c44a"}],
 "annotations":[f"{mtp[w]/base[w]:.1f}x" for w in cats],
 "footer":"WITCHEER",
}
open(sys.argv[2],"w").write(json.dumps(data,indent=2))
print(json.dumps(data))
PY
echo "[wl] wrote $OUT" >&2
