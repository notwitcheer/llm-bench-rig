#!/bin/bash
# CPU PASS-path proof of item 94 (no GPU, no Donald, no model): two mock engines (stock slower, QSA faster), REAL Hermes sessions with the real docs prefill and the 24 frozen questions answered from the key file by the mock,
# then rows, probe, recounts, chew + race render, card, concat, summary. MOCK_QA_WRONG=1 makes the mock answer every second question wrongly (the score must show it).
set -uo pipefail
H=/home/witcheer/voxel-stage/qsa94/h
RUN=/home/witcheer/voxelbench/runs/qsa94-stubtest
[ -d "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
for spec in "qm:8298:${QTPS:-190}" "sm:8299:${STPS:-85}"; do n=${spec%%:*}; r=${spec#*:}; port=${r%%:*}; tps=${r#*:}
  tmux new-session -d -s c94$n "MOCK_FLAVOUR=llama MOCK_MODEL=q27 MOCK_SCALE=${MSCALE:-1.0} MOCK_TPS=$tps MOCK_QA=$H/qa/keys/keys.json MOCK_QA_WRONG=${MOCK_QA_WRONG:-0} MOCK_PREFILL_S=${PREFILL_S:-6} python3 $H/c94_mock.py $port > /tmp/c94$n.log 2>&1"; done
sleep 2; curl -s localhost:8298/health | grep -qi ok && curl -s localhost:8299/health | grep -qi ok || { echo "mocks did not start"; tmux kill-session -t =c94qm; tmux kill-session -t =c94sm; exit 1; }
C94_STUB_QSA=http://127.0.0.1:8298 C94_STUB_STOCK=http://127.0.0.1:8299 C94_SLUG=qsa94-stubtest C94_DEPTHS="0 64000 128000 255000" C94_SEEDS="11 12" C94_PROBE_N=5 C94_NQ=24 C94_WALL_MIN=6 C94_QUIET=8 C94_TURN_QUIET=4 MANIFEST_OVERRIDE=/tmp/stub94-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) \
  bash /home/witcheer/voxel-stage/94-qsa-255k.sh; rc=$?
tmux kill-session -t =c94qm 2>/dev/null; tmux kill-session -t =c94sm 2>/dev/null
echo "item rc=$rc"
