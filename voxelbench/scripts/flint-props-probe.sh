#!/bin/bash
# flint-props-probe.sh: CPU-only, GPU hidden (CUDA_VISIBLE_DEVICES=""), no warmup: what sampling defaults does the flint build
# apply to the real Flint GGUF with pass-1 flags (none) vs pass-2 flags? reads /props only, no inference. (2026-10-22, mercury)
BIN=/home/witcheer/llama.cpp-flint/build/bin/llama-server
M=/home/witcheer/models/flint/flint-UD-Q4_K_XL.gguf
free -g | head -2
for arm in pass1-noflags pass2-maker; do
  if [ $arm = pass1-noflags ]; then F=""; else F="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"; fi
  CUDA_VISIBLE_DEVICES="" $BIN -m $M -ngl 0 -c 4096 --no-warmup -nr -t 4 --host 127.0.0.1 --port 8199 --parallel 1 $F > ~/flint-props-$arm.log 2>&1 &
  P=$!
  for i in $(seq 1 120); do curl -s 127.0.0.1:8199/health | grep -qi ok && break; kill -0 $P 2>/dev/null || break; sleep 1; done
  echo "$arm: $(curl -s 127.0.0.1:8199/props | python3 -c "import json,sys;d=json.load(sys.stdin);g=d.get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')})" 2>&1)"
  kill $P; wait $P 2>/dev/null; sleep 2
done
echo END
