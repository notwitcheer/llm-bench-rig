#!/bin/bash
# CPU PASS-path proof of item 78z (no GPU, no Donald, no model): TWO engine-shaped mocks (vLLM-flavoured at 56 tok/s, llama-flavoured at 78 tok/s) drive REAL Hermes sessions (own HERMES_HOMEs)
# through the counting proxy: 2 takes per file, tool-call smoke, gates, race render, results segment, end card, final concat, contact sheet, summary. ~6 min.
# usage (capsule): bash ~/voxelbench/race78z/test-78z-stub-e2e.sh      requires the item staged at ~/voxel-stage/85-nvfp4-race.sh
set -uo pipefail
RA=/home/witcheer/voxelbench/race78z
rm -f /tmp/r78z-mock-vllm.log /tmp/r78z-mock-llama.log
tmux new-session -d -s r78zstubv "MOCK_FLAVOUR=vllm MOCK_MODEL=qwen38-27b-nvfp4 MOCK_LOG=/tmp/r78z-mock-vllm.log MOCK_SCALE=3.0 python3 $RA/r78z_mock.py 8296 > /tmp/r78zstubv.log 2>&1"
tmux new-session -d -s r78zstubl "MOCK_FLAVOUR=llama MOCK_MODEL=Qwen3.8-27B-UD-Q4_K_M MOCK_LOG=/tmp/r78z-mock-llama.log MOCK_SCALE=3.0 python3 $RA/r78z_mock.py 8297 > /tmp/r78zstubl.log 2>&1"
sleep 1.5
[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' localhost:8296/health)" = 200 ] && curl -s localhost:8297/health | grep -qi ok || { echo "mocks did not start"; tmux kill-session -t =r78zstubv 2>/dev/null; tmux kill-session -t =r78zstubl 2>/dev/null; exit 1; }
R78Z_STUB_VLLM=http://127.0.0.1:8296 R78Z_STUB_LLAMA=http://127.0.0.1:8297 R78Z_SMOKE=1 R78Z_TAKES=2 R78Z_WALL_MIN=4 R78Z_QUIET=8 MANIFEST_OVERRIDE=/tmp/stub78z-manifest.json \
  NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) bash /home/witcheer/voxel-stage/85-nvfp4-race.sh; rc=$?
tmux kill-session -t =r78zstubv 2>/dev/null; tmux kill-session -t =r78zstubl 2>/dev/null
echo "item rc=$rc; artefacts under ~/voxelbench/runs/race78z-qwen38-nvfp4-vs-udq4km-smoke/ (move it aside before a real smoke: the item skips a slug whose summary.json exists)"
echo "what Hermes sent (one line per request, first 2 of each):"; head -2 /tmp/r78z-mock-vllm.log /tmp/r78z-mock-llama.log
