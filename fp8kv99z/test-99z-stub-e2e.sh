#!/bin/bash
# CPU PASS-path proof of item 99z (no GPU, no Donald, no model): two mocks (f16 40 tok/s, fp8 42 tok/s) play the scripted repo fix through REAL Hermes sessions behind the counting proxy;
# then gates, recount, race render, card, concat, summary. usage: bash ~/voxel-stage/kvfp8-99z-test/test-99z-stub-e2e.sh
set -uo pipefail
T=/home/witcheer/voxel-stage/ripple89-test
RUN=/home/witcheer/kvfp8-99z-runs/kvfp8-99z-smoke
[ -d "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
rm -f /tmp/kv99z-mockF.log /tmp/kv99z-mockP.log
tmux kill-session -t =kv99mockF 2>/dev/null; tmux kill-session -t =kv99mockP 2>/dev/null
tmux new-session -d -s kv99mockF "MOCK_FLAVOUR=llama MOCK_MODEL=qwen38 MOCK_LOG=/tmp/kv99z-mockF.log MOCK_SCALE=2.0 MOCK_TPS=40 MOCK_MODE=pass MOCK_REF=$T/ref python3 $T/rp_mock.py 8397 > /tmp/kv99z-mockF.out 2>&1"
tmux new-session -d -s kv99mockP "MOCK_FLAVOUR=llama MOCK_MODEL=qwen38 MOCK_LOG=/tmp/kv99z-mockP.log MOCK_SCALE=2.0 MOCK_TPS=42 MOCK_MODE=pass MOCK_REF=$T/ref python3 $T/rp_mock.py 8398 > /tmp/kv99z-mockP.out 2>&1"
sleep 2
for p in 8397 8398; do h=$(curl -s localhost:$p/health || true); [[ ${h,,} == *ok* ]] || { echo "mock $p did not start"; tmux kill-session -t =kv99mockF; tmux kill-session -t =kv99mockP; exit 1; }; done
WF_STUB_UPSTREAM_OLD=http://127.0.0.1:8397 WF_STUB_UPSTREAM_NEW=http://127.0.0.1:8398 WF_SMOKE=1 WF_TAKES=${WF_TAKES:-2} WF_WALL_MIN=4 WF_QUIET=8 MANIFEST_OVERRIDE=/tmp/stub99z-manifest.json NL_HARD_STOP_EPOCH=$(( $(date +%s) + 7200 )) \
  bash /home/witcheer/voxel-stage/kvfp8-99z-whichfile.sh; rc=$?
tmux kill-session -t =kv99mockF 2>/dev/null; tmux kill-session -t =kv99mockP 2>/dev/null
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
print("vram_mib", s.get("vram_mib"), "final_clip", s.get("final_clip"))
PYEOF
