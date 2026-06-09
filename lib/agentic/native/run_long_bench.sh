#!/usr/bin/env bash
# Long-context agentic bench for one model at one tier. Args: MODEL_GGUF SLUG TIER(32k|128k).
# Sets -c to hold the tier's document; records a VRAM wall instead of crashing. Donald-safe.
set -uo pipefail
cd ~/benchmark-rig
MODEL="$1"; SLUG="$2"; TIER="$3"; BIN=~/llama.cpp/build/bin/llama-server
case "$TIER" in
  32k)  CTX=36000;  MODE=long32k ;;
  128k) CTX=140000; MODE=long128k ;;
  *) echo "bad tier $TIER"; exit 1 ;;
esac
mkdir -p results/$SLUG
restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
"$BIN" -m "$MODEL" -ngl 999 -fa on --jinja -c "$CTX" --port 8090 --no-warmup > /tmp/long_${SLUG}_${TIER}.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 3; done
if ! curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1; then
  echo "VRAM_WALL: $SLUG did not serve at $TIER (-c $CTX). See /tmp/long_${SLUG}_${TIER}.log"; tail -4 /tmp/long_${SLUG}_${TIER}.log
  echo "{\"model\":\"$SLUG\",\"tier\":\"$TIER\",\"reach\":\"wall\"}" > results/$SLUG/agentic_longctx_${TIER}.json
  kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4; echo LONG_DONE; exit 0
fi
.venv/bin/python -m lib.agentic.native.run_native "$SLUG" "$MODE" || echo "[WARN] long run failed"
kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5
echo LONG_DONE
