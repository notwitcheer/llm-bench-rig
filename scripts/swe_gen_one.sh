#!/usr/bin/env bash
# SWE-bench anchor — single-model generation. Args: GGUF SLUG [IDS_FILE]
# Donald stopped once, restored on exit. GPU-only; grade separately.
set -uo pipefail
cd ~/benchmark-rig
GGUF="$1"; SLUG="$2"; IDS="${3:-swebench_ids_30.txt}"
BIN=~/llama.cpp/build/bin/llama-server
PY=~/swebench-env/bin/python
[ -f "$GGUF" ] || { echo "MISSING GGUF: $GGUF"; exit 3; }
[ -f "$IDS" ] || { echo "FATAL: ids $IDS missing"; exit 2; }
restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
echo "[gen] stopping Donald"; sudo -n systemctl stop llama-server.service; sleep 3
echo "==================== [$SLUG] $(date +%H:%M:%S) ===================="
"$BIN" -m "$GGUF" -ngl 999 -fa on --jinja -c 32768 --port 8090 --no-warmup > /tmp/sweb_$SLUG.log 2>&1 &
SRV=$!
ok=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
[ "$ok" = 1 ] || { echo "SERVER_NEVER_READY"; tail -5 /tmp/sweb_$SLUG.log; kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; exit 0; }
"$PY" -m lib.agentic.native.run_swebench "$SLUG" "$IDS" || echo "[$SLUG] WARN gen nonzero"
kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4
echo "SWEB_GEN_ONE_DONE $SLUG $(date +%H:%M:%S) -> predictions/$SLUG.jsonl"
