#!/usr/bin/env bash
# bench_mtp.sh — reproducible before/after for MTP self-speculative decoding on Qwen3.6-27B.
#
# WHY THIS IS NOT bench.py --speed-only:
#   The rig's speed path uses `llama-bench`, which measures raw decode with NO speculator running.
#   MTP's speedup is a serving-time, workload-dependent effect: it comes from the model drafting a
#   few tokens with its MTP heads and the full model accepting the ones it would have produced anyway.
#   You only see it during REAL generation on llama-server, where draft-acceptance rate decides the gain.
#   So this script measures generation throughput on a live llama-server, baseline vs MTP, same prompt.
#
# WHERE TO RUN: the capsule (RTX 5090 32GB). It stops the live llama-server.service for the duration
#   and ALWAYS restarts it on exit (trap), so Donald is never left offline.
#
# REQUIRES: NOPASSWD sudo for `systemctl {start,stop} <SERVICE>` (SERVICE must match the sudoers entry
#   exactly, e.g. `llama-server` not `llama-server.service`), python3, curl, nvidia-smi,
#   llama.cpp build >= 9365 (--spec-type draft-mtp), and both model files present.
set -euo pipefail

# --- config (override via env) ---------------------------------------------------------------------
HOST=127.0.0.1
PORT="${PORT:-8090}"
SERVER_BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen3.6-27B-Q6_K.gguf}"
MTP_MODEL="${MTP_MODEL:-$HOME/models/Qwen3.6-27B-MTP-Q6_K.gguf}"
SERVICE="${SERVICE:-llama-server}"      # MUST match the NOPASSWD sudoers entry exactly
CTX="${CTX:-65536}"
NGL="${NGL:-99}"
NPREDICT="${NPREDICT:-256}"             # matches the original 256-token sample
REPS="${REPS:-3}"                       # generation runs to average per config
PROMPT='Write a Python function that returns the nth Fibonacci number iteratively, then explain how it works in two short sentences.'

LOGDIR="$(mktemp -d)"
echo "[bench_mtp] logs in $LOGDIR"

# --- always put Donald's brain back, whatever happens ----------------------------------------------
restart_service() { echo "[bench_mtp] restarting $SERVICE so the agent comes back online"; sudo systemctl start "$SERVICE" || true; }
trap restart_service EXIT INT TERM

# --- helpers ---------------------------------------------------------------------------------------
wait_for_health() {  # poll /health until ready (or give up after ~150s for a cold 22GB load)
  for _ in $(seq 1 150); do
    if curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then return 0; fi
    sleep 1
  done
  echo "[bench_mtp] server failed to become healthy" >&2; return 1
}

PAYLOAD_PROMPT_JSON() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT"; }

gen_tps() {  # one timed generation; prints predicted tokens/sec from the server's own timings
  curl -s "http://$HOST:$PORT/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": $(PAYLOAD_PROMPT_JSON), \"n_predict\": $NPREDICT, \"temperature\": 0, \"cache_prompt\": false}" \
  | python3 -c 'import sys,json; print("%.1f" % json.load(sys.stdin)["timings"]["predicted_per_second"])'
}

measure() {  # $1=label  $2=model  $3..=extra server flags  ->  echoes "label|mean_tps|vram_mib|accept"
  local label="$1" model="$2"; shift 2
  local log="$LOGDIR/$label.log"
  "$SERVER_BIN" --model "$model" --n-gpu-layers "$NGL" --ctx-size "$CTX" --flash-attn on --parallel 1 \
      --host "$HOST" --port "$PORT" "$@" > "$log" 2>&1 &
  local pid=$!
  wait_for_health
  gen_tps >/dev/null  # warmup (CUDA graphs / cache settle), discarded

  local sum=0 n=0 tps
  for _ in $(seq 1 "$REPS"); do
    tps=$(gen_tps)
    sum=$(python3 -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))' "$sum" "$tps")
    n=$((n+1))
  done
  local mean; mean=$(python3 -c 'import sys; print("%.1f" % (float(sys.argv[1])/int(sys.argv[2])))' "$sum" "$n")
  local vram; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  # draft acceptance is best-effort: llama.cpp may print draft/accept stats to the server log
  local accept; accept=$(grep -ioE 'accept[a-z _-]*[0-9]+(\.[0-9]+)?[ ]*%?' "$log" | tail -1 | tr -d '|' || true)
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  sleep 3
  echo "$label|$mean|$vram|${accept:-n/a}"
}

# --- run -------------------------------------------------------------------------------------------
sudo systemctl stop "$SERVICE" || true
sleep 3

echo "[bench_mtp] baseline (no speculation)..."
BASE=$(measure baseline "$BASE_MODEL")

echo "[bench_mtp] MTP self-speculative (draft-mtp, n-max 2, q8 KV)..."
MTP=$(measure mtp "$MTP_MODEL" --spec-type draft-mtp --spec-draft-n-max 2 --cache-type-k q8_0 --cache-type-v q8_0)

# --- report ----------------------------------------------------------------------------------------
echo "RESULT_BASE=$BASE"
echo "RESULT_MTP=$MTP"
python3 - "$BASE" "$MTP" "$NPREDICT" "$REPS" <<'PY'
import sys
def row(s):
    l,t,v,a = s.split("|"); return l, float(t), v, a
bl, bt, bv, ba = row(sys.argv[1])
ml, mt, mv, ma = row(sys.argv[2])
npred, reps = sys.argv[3], sys.argv[4]
print(f"\n=== MTP speedup (Qwen3.6-27B, RTX 5090, {npred}-tok generation, mean of {reps}) ===")
print(f"{'config':<10} {'gen tok/s':>10} {'VRAM MiB':>10} {'accept':>14}")
print(f"{bl:<10} {bt:>10.1f} {bv:>10} {ba:>14}")
print(f"{ml:<10} {mt:>10.1f} {mv:>10} {ma:>14}")
print(f"\nspeedup: {mt/bt:.2f}x  ({bt:.0f} -> {mt:.0f} tok/s)")
PY
echo "[bench_mtp] done. (service will be restarted by the exit trap)"
