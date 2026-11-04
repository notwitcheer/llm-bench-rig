#!/bin/bash
# CPU PASS-path proof of item 80b (no GPU, no Donald): four REAL Hermes sessions (own HERMES_HOMEs, the v2-compact config) against four_mock.py through the counting proxy + pool guard.
# The mock reports about 1 token per 3.2 bytes of conversation (+ its decoded tokens) and simulates the pooled KV (163,840), so a short script reaches real-sized contexts and the item-80 failure is reproducible on the CPU.
# usage: bash test-80b-stub-e2e.sh [compact|control|guard]
#   compact  (default) 80b config: threshold_tokens 34000 + protect_first_n 0 + proxy guard  -> expect rule PASS, compaction events in all four homes
#   control  item-80 config (FOUR_COMPACT_TOKENS=0, guard off)             -> expect the mock pool to run dry: rule FAIL (proves the mock reproduces the failure and the rule catches it)
#   guard    guard on, compaction OFF (0)                                  -> shows what the proxy guard alone does (holds, maybe still FAIL by the wall)
# needs the item staged at ~/voxel-stage/80b-four-agents-compact.sh (or queued). ~6-8 min each, CPU only. The item skips a slug whose summary.json exists: each mode has its own slug suffix.
set -uo pipefail
MODE=${1:-compact}
FA=/home/witcheer/voxelbench/four-agents
ITEM=${ITEM:-/home/witcheer/voxel-stage/80b-four-agents-compact.sh}
case "$MODE" in
  compact) CT=34000; GP=92;;
  control) CT=0; GP=0;;
  guard)   CT=0; GP=92;;
  *) echo "mode compact|control|guard"; exit 2;;
esac
tmux new-session -d -s f80bmock "MOCK_SCALE=1.0 MOCK_TPS=${MOCK_TPS:-900} MOCK_FIRST_THINK=${MOCK_FIRST_THINK:-16000} MOCK_TOKMULT=${MOCK_TOKMULT:-1} MOCK_TURNS=${MOCK_TURNS:-40} MOCK_CAT=${MOCK_CAT:-3} MOCK_POOL=163840 python3 $FA/four_mock.py 8196 > /tmp/f80bmock-$MODE.log 2>&1"
sleep 1; curl -s localhost:8196/health | grep -qi ok || { echo "mock did not start"; exit 1; }
FOUR_STUB_UPSTREAM=http://127.0.0.1:8196 FOUR_SMOKE=1 FOUR_WALL_MIN=${FOUR_WALL_MIN:-6} FOUR_SLUG_SUFFIX=-t$MODE FOUR_COMPACT_TOKENS=$CT FOUR_POOL_GUARD=$GP \
  MANIFEST_OVERRIDE=/tmp/stub80b-$MODE-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) bash "$ITEM"; rc=$?
curl -s localhost:8196/_mock > /tmp/f80bmock-$MODE-state.json
tmux kill-session -t =f80bmock 2>/dev/null
echo "item rc=$rc mode=$MODE; run dir ~/voxelbench/runs/four80b-qwen38-q6k-par4-compact-t$MODE-smoke/ ; mock state /tmp/f80bmock-$MODE-state.json"
