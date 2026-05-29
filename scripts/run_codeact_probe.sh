#!/usr/bin/env bash
cd /home/witcheer/benchmark-rig
pkill -f llama-server 2>/dev/null
sleep 1
MP="/home/witcheer/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
/home/witcheer/llama.cpp/build/bin/llama-server -m "$MP" --port 8090 -ngl 99 --host 127.0.0.1 --ctx-size 16384 > /tmp/srv.log 2>&1 &
for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:8090/health >/dev/null && break
  sleep 2
done
echo "=== health wait done ==="
.venv/bin/python scripts/probe_codeact.py
pkill -f llama-server 2>/dev/null
echo "=== probe done ==="
