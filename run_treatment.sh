#!/bin/bash
# 5090 Treatment runner: stop Hermes server, run benchmarks, ALWAYS restore.
set -euo pipefail
cd ~/benchmark-rig
source venv/bin/activate
export PYTHONUNBUFFERED=1
PORT=8090

restore_hermes() {
  echo "[guard] restoring llama-server.service ..."
  sudo systemctl start llama-server.service || true
}
trap restore_hermes EXIT

echo "[guard] stopping Hermes llama-server.service ..."
sudo systemctl stop llama-server.service || true

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
