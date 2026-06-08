#!/usr/bin/env bash
# Measure decode tok/s with vs without the MTP drafter (speculative decoding).
# Args: MAIN_GGUF DRAFT_GGUF.  Reads /completion timings.predicted_per_second.
set -uo pipefail
MAIN="$1"; DRAFT="$2"
BIN=~/llama.cpp/build/bin/llama-server

cat > /tmp/mtp_req.json <<'JSON'
{"prompt":"Write a detailed 400-word technical explanation of how a modern CPU instruction pipeline works, covering fetch, decode, execute, memory, and writeback stages, plus hazards and branch prediction.","n_predict":400,"temperature":0,"cache_prompt":false}
JSON

measure() {
  local label="$1"; shift
  pkill -9 -x llama-server 2>/dev/null; sleep 3
  "$BIN" -m "$MAIN" -ngl 999 -fa on --port 8090 --no-warmup "$@" > /tmp/mtp_$label.log 2>&1 &
  local pid=$!
  local up=""
  for i in $(seq 1 80); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && { up=1; break; }; sleep 3; done
  if [ -z "$up" ]; then echo "$label: SERVER_FAILED"; tail -4 /tmp/mtp_$label.log; kill -9 $pid 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 3; return; fi
  local out=$(curl -s http://127.0.0.1:8090/completion -H 'Content-Type: application/json' --data @/tmp/mtp_req.json \
    | python3 -c "import sys,json; d=json.load(sys.stdin)['timings']; print(round(d['predicted_per_second'],2), 'n='+str(d.get('predicted_n')))" 2>/dev/null)
  echo "$label: decode_tps=$out"
  kill -9 $pid 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4
}

measure baseline
measure mtp --model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4
echo MTP_SPEED_DONE
