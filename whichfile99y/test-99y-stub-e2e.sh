#!/bin/bash
# CPU PASS-path proof of item 99y (no GPU, no Donald, no model): two mocks (old 40 tok/s, new 42 tok/s) play the scripted repo fix against REAL Hermes sessions behind the counting proxy;
# then gates, recount, race render, card, concat, summary. usage: bash ~/voxel-stage/whichfile99y-test/test-99y-stub-e2e.sh   (WF_TAKES=2 default)
set -uo pipefail
T=/home/witcheer/voxel-stage/ripple89-test
RUN=/home/witcheer/whichfile99y-runs/whichfile99y-unsloth-v31-smoke
[ -d "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
rm -f /tmp/wf99y-mockO.log /tmp/wf99y-mockN.log
tmux kill-session -t =wf99mockO 2>/dev/null; tmux kill-session -t =wf99mockN 2>/dev/null
tmux new-session -d -s wf99mockO "MOCK_FLAVOUR=llama MOCK_MODEL=qwen38 MOCK_LOG=/tmp/wf99y-mockO.log MOCK_SCALE=2.0 MOCK_TPS=40 MOCK_MODE=pass MOCK_REF=$T/ref python3 $T/rp_mock.py 8297 > /tmp/wf99y-mockO.out 2>&1"
tmux new-session -d -s wf99mockN "MOCK_FLAVOUR=llama MOCK_MODEL=qwen38 MOCK_LOG=/tmp/wf99y-mockN.log MOCK_SCALE=2.0 MOCK_TPS=42 MOCK_MODE=pass MOCK_REF=$T/ref python3 $T/rp_mock.py 8298 > /tmp/wf99y-mockN.out 2>&1"
sleep 2
for p in 8297 8298; do h=$(curl -s localhost:$p/health || true); [[ ${h,,} == *ok* ]] || { echo "mock $p did not start"; tmux kill-session -t =wf99mockO; tmux kill-session -t =wf99mockN; exit 1; }; done
WF_STUB_UPSTREAM_OLD=http://127.0.0.1:8297 WF_STUB_UPSTREAM_NEW=http://127.0.0.1:8298 WF_SMOKE=1 WF_TAKES=${WF_TAKES:-2} WF_WALL_MIN=4 WF_QUIET=8 MANIFEST_OVERRIDE=/tmp/stub99y-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) \
  bash /home/witcheer/voxel-stage/99y-unsloth-v31-whichfile.sh; rc=$?
tmux kill-session -t =wf99mockO 2>/dev/null; tmux kill-session -t =wf99mockN 2>/dev/null
echo "item rc=$rc"
python3 - "$RUN" <<'PYEOF'
import json, sys
run = sys.argv[1]
try:
    s = json.load(open(run + "/summary.json"))
except Exception:
    s = json.load(open(run + "/summary.pre.json"))
print("verdict", s["verdict"], "line1", s.get("line1"))
print("arms", json.dumps(s["arms"]))
for k, t in sorted(s["takes"].items()):
    print(k, {x: t.get(x) for x in ("end", "finish_s", "tool_calls", "clean", "pass", "gate_pass", "tests_touched", "tokens_server", "decode_tok_s_engine")})
print("file_bytes", s.get("file_bytes"), "final_clip", s.get("final_clip"))
PYEOF
