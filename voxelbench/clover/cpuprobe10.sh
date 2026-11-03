#!/bin/bash
# CPU probe 10: LONG-TURN acceptance. one big write_file (a ~120-line self-contained HTML canvas page, the shape of a Mont-Saint-Michel build turn) with presence 0.9 + repeat 1.05 as served (A) vs both off per request (B). max_tokens 3000. port 8195, CUDA hidden.
set -u
BIN=/home/witcheer/llama.cpp-clover-pr/build/bin/llama-server
M=/home/witcheer/models/clover-35b/Clover-35B-A3B-Preview-Q4_K_M.gguf
OUT=/home/witcheer/voxel-stage/clover/cpuprobe10; mkdir -p $OUT
CUDA_VISIBLE_DEVICES= $BIN --model $M --host 127.0.0.1 --port 8195 -ngl 0 --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 --flash-attn on --jinja --no-warmup -nr --metrics -t 10 \
  --temp 1.0 --top-k 0 --top-p 1.0 --min-p 0 --presence-penalty 0.9 --repeat-penalty 1.05 --reasoning-budget 600 --reasoning-budget-message "Thinking budget reached. Stop planning and write the files now with your tools." \
  --spec-type draft-self --spec-draft-n-max 4 > $OUT/server.log 2>&1 &
pid=$!
for i in $(seq 1 300); do curl -s 127.0.0.1:8195/health | grep -qi ok && break; kill -0 $pid 2>/dev/null || { echo died; exit 1; }; sleep 2; done
echo "$(date +%T) up"
export MT=3000 TASK="Write /tmp/clover-loop/scene.html: ONE self-contained HTML file with a <canvas>, plain JavaScript (no libraries), that draws an isometric voxel island: a 24x24 height-map built from two sine waves, each column drawn as stacked cubes with three shaded faces, a stone castle (at least 40 cubes) on the highest point, and a slow camera rotation animated with requestAnimationFrame. Comment each function. Use ONE write_file call with the complete file, then say done."
echo "== B penalties off (first, warm)"; SAMP='{"presence_penalty":0,"repeat_penalty":1.0}' timeout 420 python3 /home/witcheer/voxel-stage/clover/loop2.py 8195 $OUT/loop-B.json 1 x
grep -E "draft acceptance" $OUT/server.log | tail -3 | cut -c1-200
echo "== A penalties as served"; SAMP='{}' timeout 600 python3 /home/witcheer/voxel-stage/clover/loop2.py 8195 $OUT/loop-A.json 1 x
grep -E "draft acceptance" $OUT/server.log | tail -3 | cut -c1-200
kill $pid; sleep 4; kill -9 $pid 2>/dev/null
echo PROBE10-DONE
