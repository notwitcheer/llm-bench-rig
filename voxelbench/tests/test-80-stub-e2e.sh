#!/bin/bash
# CPU PASS-path proof of item 80 (no GPU, no Donald): four real Hermes sessions (own HERMES_HOMEs) against four_mock.py through the counting proxy,
# then gates, clip, mosaic, summary. ~5 min. usage: bash tests/test-80-stub-e2e.sh   (capsule; the item must be staged at ~/voxel-stage/80-four-agents-one-gpu.sh)
set -uo pipefail
FA=/home/witcheer/voxelbench/four-agents
tmux new-session -d -s f80stubmock "MOCK_SCALE=3.0 python3 $FA/four_mock.py 8196 > /tmp/f80stubmock.log 2>&1"
sleep 1; curl -s localhost:8196/health | grep -qi ok || { echo "mock did not start"; exit 1; }
FOUR_STUB_UPSTREAM=http://127.0.0.1:8196 FOUR_SMOKE=1 FOUR_WALL_MIN=5 MANIFEST_OVERRIDE=/tmp/stub80-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) \
  bash /home/witcheer/voxel-stage/80-four-agents-one-gpu.sh; rc=$?
tmux kill-session -t =f80stubmock 2>/dev/null
echo "item rc=$rc; artefacts under ~/voxelbench/runs/four80-qwen38-q6k-par4-smoke/ (move it aside before a real smoke: the item skips a slug whose summary.json exists)"
