#!/usr/bin/env bash
# Run the native agentic bench for a model. Args: MODEL_GGUF SLUG. Donald-safe:
# stops llama-server.service, serves the subject on :8090 with --jinja tool-calling,
# runs the bench, then ALWAYS restarts Donald via the EXIT trap.
set -uo pipefail
cd ~/benchmark-rig
MODEL="$1"; SLUG="$2"; BIN=~/llama.cpp/build/bin/llama-server
restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
"$BIN" -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --port 8090 --no-warmup > /tmp/native_$SLUG.log 2>&1 &
SRV=$!
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 3; done
.venv/bin/python -m lib.agentic.native.run_native "$SLUG" || echo "[WARN] run failed"
kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4
echo NATIVE_DONE
