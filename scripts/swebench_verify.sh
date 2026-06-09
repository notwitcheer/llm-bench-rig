#!/usr/bin/env bash
# SWE-bench anchor for one model. Args: MODEL_GGUF SLUG IDS_FILE.
# Phase 1 (generation): serve the model, run the agent loop over the pinned instances.
# Donald-safe (restores llama-server.service on exit). Grading (phase 2) is run separately.
set -uo pipefail
cd ~/benchmark-rig
MODEL="$1"; SLUG="$2"; IDS="$3"; BIN=~/llama.cpp/build/bin/llama-server
restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
"$BIN" -m "$MODEL" -ngl 999 -fa on --jinja -c 32768 --port 8090 --no-warmup > /tmp/sweb_$SLUG.log 2>&1 &
SRV=$!
for i in $(seq 1 100); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 3; done
~/swebench-env/bin/python -m lib.agentic.native.run_swebench "$SLUG" "$IDS" || echo "[WARN] generation failed"
kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4
echo SWEB_GEN_DONE
