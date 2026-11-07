#!/bin/bash
# CPU PASS-path proof of item 93 (no GPU, no Donald, no model, no download): two engine-shaped mocks (llama flavour, basalt flavour) play a scripted repo fix against REAL Hermes sessions (own HERMES_HOMEs)
# behind the counting proxy; then rows, ladder stub, gates, recounts, race render, result card, concat, summary. MOCK_MODE=pass|partial.
set -uo pipefail
T=/home/witcheer/voxel-stage/ripple89-test; M=/home/witcheer/voxel-stage/basalt93/h
RUN=/home/witcheer/voxelbench/runs/basalt93-stubtest
[ -d "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
tmux new-session -d -s b93mockl "MOCK_FLAVOUR=llama MOCK_MODEL=fn MOCK_SCALE=${MSCALE:-2.0} MOCK_TPS=${LTPS:-62} MOCK_MODE=${MOCK_MODE:-pass} MOCK_REF=$T/ref python3 $M/b93_mock.py 8296 > /tmp/b93mockl.log 2>&1"
tmux new-session -d -s b93mockb "MOCK_FLAVOUR=basalt MOCK_MODEL=fn MOCK_SCALE=${MSCALE:-2.0} MOCK_TPS=${BTPS:-190} MOCK_MODE=${MOCK_MODE:-pass} MOCK_REF=$T/ref python3 $M/b93_mock.py 8297 > /tmp/b93mockb.log 2>&1"
sleep 2; curl -s localhost:8296/health | grep -qi ok && curl -s localhost:8297/health | grep -qi ok || { echo "mocks did not start"; tmux kill-session -t =b93mockl; tmux kill-session -t =b93mockb; exit 1; }
B93_STUB_LLAMA=http://127.0.0.1:8296 B93_STUB_BASALT=http://127.0.0.1:8297 B93_SLUG=basalt93-stubtest B93_TAKES=2 B93_REPS=3 B93_WALL_MIN=4 B93_QUIET=8 MANIFEST_OVERRIDE=/tmp/stub93-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) \
  bash /home/witcheer/voxel-stage/93-basalt-race.sh; rc=$?
tmux kill-session -t =b93mockl 2>/dev/null; tmux kill-session -t =b93mockb 2>/dev/null
echo "item rc=$rc"
ls $RUN $RUN/out 2>/dev/null | head -40
