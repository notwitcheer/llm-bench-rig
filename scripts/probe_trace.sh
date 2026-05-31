#!/usr/bin/env bash
# Probe: run one multi-step Hermes task with FULL output capture to discover
# what trajectory data (tool calls, steps, errors) is recoverable for metrics.
MP="/home/witcheer/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q6_K.gguf"
pkill -f llama-server 2>/dev/null; sleep 1
/home/witcheer/llama.cpp/build/bin/llama-server -m "$MP" --port 8090 -ngl 99 --host 127.0.0.1 --ctx-size 65536 > /tmp/srv.log 2>&1 &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8090/health >/dev/null && break; sleep 2; done
mkdir -p /tmp/phaseb; printf '12\n30\n8\n50\n' > /tmp/phaseb/input.csv; rm -f /tmp/phaseb/verdict.txt
SESS_BEFORE=$(ls -1 ~/.hermes/sessions/ 2>/dev/null | wc -l)
cd ~/hermes-src
timeout 220 .venv/bin/hermes -z "Read /tmp/phaseb/input.csv (one integer per line). Sum the integers. If the sum is greater than 50, write the single word HIGH to /tmp/phaseb/verdict.txt, otherwise write LOW. Then read /tmp/phaseb/verdict.txt back and tell me what it contains." --yolo < /dev/null > /tmp/htrace.out 2> /tmp/htrace.err
pkill -f llama-server 2>/dev/null
echo "PROBE-DONE"
echo "=== stdout size: $(wc -c < /tmp/htrace.out)  stderr size: $(wc -c < /tmp/htrace.err) ==="
echo "=== newest session files ==="
ls -lt ~/.hermes/sessions/ | head -4
