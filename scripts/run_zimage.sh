#!/bin/bash
# Z-Image-Turbo bench runner (capsule). Drains Donald (llama-server) for the GPU
# window, runs the synth adapter, and ALWAYS restarts Donald on exit via a trap
# — the model window must never leave Donald down. Mirrors run_treatment.sh.
set +e
mkdir -p ~/zimage-out

restore() {
  echo "=== restoring Donald (start llama-server) ==="
  sudo systemctl start llama-server.service
  sleep 3
  systemctl is-active llama-server.service
}
trap restore EXIT

echo "=== draining Donald (stop llama-server) ==="
sudo systemctl stop llama-server.service
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "vram used: ${used}MiB"
  [ "$used" -lt 4000 ] && { echo "drained."; break; }
  sleep 1
done

echo "=== running Z-Image-Turbo synth ==="
~/unsloth-env/bin/python ~/zimage-work/zimage_synth.py \
  --model /home/witcheer/models/z-image-turbo \
  --prompts ~/zimage-work/prompts.json \
  --resolutions 512,1024,1536,2048 --steps 8 \
  --step-sweep 4,8,9,16 --sweep-resolution 1024 \
  --seed 42 --out-dir ~/zimage-out \
  --synth-json ~/zimage-out/synth.json --save-images
echo "=== synth exit: $? ==="
