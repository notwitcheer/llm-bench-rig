#!/bin/bash
# ab-nomtp.sh: one short GPU probe for item 80 (staging only, ~3 min card): the same server flags as the item at the chosen rung, with and without MTP,
# then four_probe.py (solo + four at once) on each. Own pids only, Donald drained with sudo -n systemctl stop, restored with start, memory polled under 1 GiB first.
# run by hand after a live ownership check; never inside ~/idle-queue.
set -uo pipefail
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
MODEL=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
OUT=/home/witcheer/voxel-stage/four80/ab; mkdir -p $OUT
PORT=8190
SRV=""
killsrv() { { [ -n "$SRV" ] && kill -9 "$SRV" && wait "$SRV"; } 2>/dev/null; SRV=""; }
back() { killsrv; sleep 3; sudo -n systemctl start llama-server.service; for i in $(seq 1 60); do curl -s -m 5 localhost:8090/health | grep -qi ok && { echo "Donald healthy again ~$((i*5)) s"; return; }; sleep 5; done; echo "WARNING Donald not healthy"; }
trap back EXIT
sudo -n systemctl stop llama-server.service; sleep 5
for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && break; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || { echo "card not free (${v} MiB): abort"; exit 1; }
for leg in "mtp:163840:--spec-type draft-mtp --spec-draft-n-max 2" "nomtp:163840:" "mtp196k:196608:--spec-type draft-mtp --spec-draft-n-max 2"; do
  name=${leg%%:*}; rest=${leg#*:}; ctx=${rest%%:*}; extra=${rest#*:}
  echo "== leg $name ctx $ctx extra '$extra'"
  # shellcheck disable=SC2086
  $BIN --model $MODEL --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size $ctx --parallel 4 --kv-unified --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --alias Qwen3.8-27B-Q6_K.gguf --metrics --reasoning-budget 16384 --reasoning-budget-message "x" --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 $extra > $OUT/server-$name.log 2>&1 &
  SRV=$!
  up=0; for i in $(seq 1 90); do curl -s -m 3 127.0.0.1:$PORT/health | grep -qi ok && { up=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  [ "$up" = 1 ] || { echo "leg $name did not come up"; killsrv; sleep 4; continue; }
  echo "load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/vram-$name.csv; sleep 1; done ) & VP=$!
  python3 /home/witcheer/voxelbench/four-agents/four_probe.py http://127.0.0.1:$PORT Qwen3.8-27B-Q6_K.gguf $OUT/probe-$name.json 2>&1 | tail -1
  kill $VP 2>/dev/null; wait $VP 2>/dev/null
  echo "vram peak during probe: $(sort -n $OUT/vram-$name.csv | tail -1) MiB"
  killsrv; sleep 5
done
