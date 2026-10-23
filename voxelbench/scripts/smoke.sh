#!/bin/bash
# voxel driver smoke (2026-10-21 daytime, <15 min of card): tiny goal, 3 turns, 8 min wall, Donald drained + restored.
set -uo pipefail
V=/home/witcheer/voxelbench
SL=${1:-smoke2}
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 3; sudo -n systemctl start llama-server.service; echo "[smoke] $(date +%T) Donald restored: $(systemctl is-active llama-server.service)"; }
trap restore EXIT
echo "[smoke] $(date +%T) drain"; sudo -n systemctl stop llama-server.service; sleep 3
/home/witcheer/llama.cpp-donald/build/bin/llama-server --model /home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf \
  --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size 131072 --parallel 1 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 2 --cache-type-k q8_0 --cache-type-v q8_0 --alias Qwen3.8-27B-Q6_K.gguf \
  > $V/smoke-server.log 2>&1 &
for i in $(seq 1 90); do curl -s 127.0.0.1:8090/health | grep -qi ok && break; sleep 2; done
echo "[smoke] $(date +%T) server: $(curl -s 127.0.0.1:8090/health) vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
cat > $V/smoke-prompt.txt <<'EOF'
Create index.html in the current working directory: a single HTML file showing a small animated voxel scene (a few coloured cubes rotating) with plain WebGL, no external libraries. Verify your work by reading the file back.
EOF
# never delete a bench profile: the v0.22 host gateway holds every profile's state.db open (deleted-WAL park). unique slug per run.
bash $V/scripts/run_agent.sh $SL Qwen3.8-27B-Q6_K.gguf 131072 8 3 $V/smoke-prompt.txt
echo "[smoke] $(date +%T) driver: $(cat $V/runs/$SL/driver.json)"
ls -la $V/runs/$SL/work/
H=$(ls $V/runs/$SL/work/*.html 2>/dev/null | head -1)
[ -n "$H" ] && timeout 120 /home/witcheer/benchmark-rig/.venv/bin/python $V/scripts/gate.py "$H" $V/runs/$SL/gate --video-seconds 12
echo "[smoke] $(date +%T) gate rc=$?"
