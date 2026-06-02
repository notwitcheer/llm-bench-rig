#!/bin/bash
# 5090 Treatment runner: stop Hermes server, run benchmarks, ALWAYS restore.
set -euo pipefail
cd ~/benchmark-rig
source venv/bin/activate
export PYTHONUNBUFFERED=1
PORT=8090

restore_hermes() {
  echo "[guard] cleaning up any stray llama-server on port $PORT before restore ..."
  # Kill orphaned bench-launched servers still holding the port (NOT the systemd unit, which is already stopped).
  STRAY_PIDS=$(ss -ltnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  if [ -n "$STRAY_PIDS" ]; then
    echo "[guard] killing stray PID(s): $STRAY_PIDS"
    # shellcheck disable=SC2086
    kill $STRAY_PIDS 2>/dev/null || true
    sleep 3
    kill -9 $STRAY_PIDS 2>/dev/null || true
  fi
  echo "[guard] restoring llama-server.service ..."
  if ! sudo -n systemctl start llama-server.service; then
    echo "[guard] ERROR: failed to restart llama-server.service — Hermes may be DOWN. Investigate manually." >&2
  fi
}
trap restore_hermes EXIT

echo "[guard] stopping Hermes llama-server.service ..."
sudo -n systemctl stop llama-server.service || true

echo "[guard] waiting for port $PORT free + VRAM drained (<4GB) ..."
for i in $(seq 1 120); do
  PORT_BUSY=$(ss -ltn "sport = :$PORT" | tail -n +2 | wc -l)
  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$PORT_BUSY" -eq 0 ] && [ "$VRAM" -lt 4000 ]; then
    echo "[guard] clear (port free, VRAM ${VRAM}MiB)"; break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then echo "[guard] TIMEOUT waiting for clear state" >&2; exit 1; fi
done

# Models passed as args (full GGUF paths). Each runs through bench.py.
for MODEL in "$@"; do
  NAME=$(basename "$MODEL" .gguf)
  echo "━━━ benchmarking $NAME ━━━"
  if python3 bench.py "$MODEL"; then echo "  ✓ $NAME"; else echo "  ✗ $NAME FAILED — continuing" >&2; fi
done
echo "[done] treatment run complete"
