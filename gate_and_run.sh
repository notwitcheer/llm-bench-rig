#!/usr/bin/env bash
# Gate + run one model through the native agentic harness in a SINGLE serving.
# Args: MODEL_GGUF  SLUG  [EXTRA_LLAMA_ARGS...]
# Donald-safe: restarts llama-server.service on exit. Skips the harness if the
# tool-calling gate fails (prints GATE_FAIL) so we never publish an ungated run.
set -uo pipefail
cd ~/benchmark-rig
MODEL="$1"; SLUG="$2"; shift 2; EXTRA=("$@")
BIN=~/llama.cpp/build/bin/llama-server
restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
"$BIN" -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --port 8090 --no-warmup "${EXTRA[@]}" > /tmp/gr_$SLUG.log 2>&1 &
SRV=$!
for i in $(seq 1 100); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 3; done
if ! curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1; then
  echo "SERVER_NEVER_READY (see /tmp/gr_$SLUG.log)"; tail -5 /tmp/gr_$SLUG.log
  kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 4; echo GR_DONE; exit 0
fi
echo "=== HARD GATE: $SLUG native tool_calls ==="
GATE=$(curl -s http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 21*2? Use the calc tool."}],"tools":[{"type":"function","function":{"name":"calc","description":"evaluate a numeric expression","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}}],"tool_choice":"auto","temperature":0}' \
  | .venv/bin/python -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; tc=m.get("tool_calls"); print("PASS" if tc and tc[0]["function"]["name"]=="calc" else "FAIL"); print("tool_calls:", json.dumps(tc))')
echo "$GATE"
if echo "$GATE" | grep -q "^PASS"; then
  echo "=== GATE PASS -> running harness for $SLUG ==="
  .venv/bin/python -m lib.agentic.native.run_native "$SLUG" || echo "[WARN] $SLUG run failed"
else
  echo "=== GATE FAIL -> skipping harness for $SLUG (needs a parse shim) ==="
fi
kill -9 $SRV 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5
echo GR_DONE
