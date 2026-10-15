#!/bin/bash
# toolcall-probe.sh: toolcall-v1 step-1 GPU probe (2026-10-15, mercury; operator go on the design doc
# 99-personal/benchmark-rig/drafts/2026-10-15-toolcall-slice-design.md). Daytime, under 30 min of card time
# (electricity rule allows short runs in heures pleines). Two legs, think-on 16k, raw /completion:
#   q6k  : Qwen3.8-27B Q6_K on the rig mainline binary (~/llama.cpp-nl, build 10371 5d16e81dd)
#   ptq1 : Ternary-Bonsai-2-27B PTQ1_0 on the PrismML fork (~/llama.cpp-prism 4fbda12)
# server flags = the board's (lib/quality.py build_server_command: -m --port --n-gpu-layers 99 --host --jinja
# --ctx-size) with ctx 40960 instead of 24576 so the S5 multi-turn items can show their real prompt growth
# (single-turn prompts max 6140 + 16384 fit either way; n_ctx does not change greedy outputs).
# per leg: /props, /tokenize parity on all 1998 frozen prompts, 5 items per S1..S4, S5 entries until the
# leg budget runs out. Donald drained with an EXIT-trap restore. results: ~/benchmark-rig/results/probe-toolcall-2026-10-15/
set -uo pipefail
NL_TAG=tcprobe
MANIFEST=/home/witcheer/toolcall-probe-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
T=/home/witcheer/toolcall-v1
OUT=/home/witcheer/benchmark-rig/results/probe-toolcall-2026-10-15
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
Q6K=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
PTQ1=/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf
PORT=8097
CTX=40960
# hard stop: 13:20 local today (operator window ends 13:30) and never more than 29 min after start
STOP_LOCAL=$(date -d "today 13:20" +%s)
CAP=$(( $(date +%s) + 29*60 ))
DEADLINE=$(( STOP_LOCAL < CAP ? STOP_LOCAL : CAP ))
T_START=$(date +%s)

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$MAIN" "$PRISM" "$Q6K" "$PTQ1" "$T/.venv/bin/python" "$T/code/driver.py" "$T/frozen/toolcall-v1/manifest.json" \
  "$T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz" "$T/artifacts/probe_ids.json" "$T/qwen38-tokenizer/tokenizer.json"
nl_check_disk 20
( cd $T/frozen/toolcall-v1 && sha256sum -c --quiet FROZEN.sha256 ) && nl_ok "frozen list sha256 intact" || nl_fail "frozen list changed"
PV=$($PRISM --version 2>&1 || true); case "$PV" in *4fbda12*) nl_ok "prism build 4fbda12";; *) nl_fail "prism build changed: $PV";; esac
MV=$($MAIN --version 2>&1 || true); case "$MV" in *5d16e81*) nl_ok "mainline build 5d16e81";; *) nl_fail "mainline build changed: $MV";; esac
LIVE=$(tmux ls 2>/dev/null | cut -d: -f1 | grep -E '^(night|idleq|recipe|budget|fnext|lcbf|lcgaps|gpqa|secondtier)' || true)
[ -z "$LIVE" ] && nl_ok "no GPU-owning tmux session" || nl_fail "GPU-owning tmux live: $LIVE"
LEFT=$(( (DEADLINE - $(date +%s)) / 60 ))
[ "$LEFT" -ge 20 ] && nl_ok "window: $LEFT min until hard stop $(date -d "@$DEADLINE" +%H:%M)" || nl_fail "only $LEFT min before the 13:20 hard stop"
mkdir -p "$OUT"
[ -e "$OUT/q6k/summary.json" ] && nl_fail "results exist already: $OUT/q6k (move aside first)"
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init q6k ptq1 cross_binary; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
  for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:8090/health | grep -qi ok && { nl_say "Donald proxy health OK"; break; }; sleep 2; done; }
trap restore EXIT
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done

leg() { # <name> <bin> <gguf> <budget_s> <bin label>
  local name=$1 bin=$2 gguf=$3 budget=$4 blabel=$5 t0 rc
  mkdir -p "$OUT/$name"; t0=$(date +%s); nl_mark "$name" running
  local left=$(( DEADLINE - $(date +%s) - 90 ))
  [ "$budget" -gt "$left" ] && budget=$left
  [ "$budget" -gt 240 ] || { nl_say "  $name: window too short ($left s), skipped"; nl_mark "$name" failed "window"; return 1; }
  nl_say "=== $name: $(basename "$gguf") on $blabel, ctx $CTX, budget ${budget}s ==="
  "$bin" -m "$gguf" --port $PORT --n-gpu-layers 99 --host 127.0.0.1 --jinja --ctx-size $CTX > "$OUT/$name/server.log" 2>&1 &
  local sp=$!
  for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && break; kill -0 $sp 2>/dev/null || break; sleep 2; done
  curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok || { nl_say "  $name: server did not come up (tail server.log)"; tail -5 "$OUT/$name/server.log"; kill $sp 2>/dev/null; nl_mark "$name" failed "server load"; return 1; }
  nl_say "  server up in $(( $(date +%s) - t0 ))s, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  ( nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader -l 5 > "$OUT/$name/gpu-samples.csv" 2>/dev/null ) &
  local gp=$!
  cd $T/code
  timeout $(( budget + 120 )) ../.venv/bin/python -u driver.py --server http://127.0.0.1:$PORT --label "$name" \
      --ids $T/artifacts/probe_ids.json --manifest $T/frozen/toolcall-v1/manifest.json --tokenizer $T/qwen38-tokenizer \
      --prompts $T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz --out "$OUT/$name" --think on --max-tokens 16384 --ctx $CTX \
      --budget-s "$budget" --tokenize-all --server-bin "$blabel" > "$OUT/$name/driver.log" 2>&1
  rc=$?
  grep -v -i "warn" "$OUT/$name/driver.log" | grep -E "^\[|tokenize_parity|Traceback|Error" | tail -40
  kill $gp 2>/dev/null; kill $sp 2>/dev/null; sleep 3; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if nl_judge_artifact "$name" "$rc" "$OUT/$name/summary.json" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $name done in $(( $(date +%s)-t0 ))s"; else nl_say "  $name FAILED rc=$rc"; fi
}

leg q6k "$MAIN" "$Q6K" 780 "ggml-org/llama.cpp mainline build 10371 (5d16e81dd), CUDA, sm_120"
leg ptq1 "$PRISM" "$PTQ1" 600 "PrismML-Eng/llama.cpp prism 4fbda12 (CUDA 12.8, sm_120)"

/usr/bin/python3 - "$OUT" <<'PY' && nl_mark cross_binary done || nl_mark cross_binary failed
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
a = {(r["id"], r["think"]): r for r in json.load(open(o / "q6k/tokenize_parity.json"))}
b = {(r["id"], r["think"]): r for r in json.load(open(o / "ptq1/tokenize_parity.json"))}
k = sorted(set(a) & set(b))
diff = [x for x in k if a[x]["server_ids_sha256"] != b[x]["server_ids_sha256"]]
res = {"prompts_compared": len(k), "token_ids_differ_between_binaries": len(diff), "differing": diff[:20],
       "q6k_vs_hf_mismatch": sum(1 for x in k if not a[x]["identical"]), "ptq1_vs_hf_mismatch": sum(1 for x in k if not b[x]["identical"])}
json.dump(res, open(o / "cross_binary_tokenize.json", "w"), indent=1)
print("[tcprobe] cross-binary tokenize:", json.dumps(res))
PY
nl_say "probe finished, $(( ($(date +%s) - T_START) / 60 )) min since start (drain to last leg end)"
exit 0
