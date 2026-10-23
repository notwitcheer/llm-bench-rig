#!/bin/bash
# day0-voxel.sh: release-day "first look" voxel run (2026-10-21, mercury, GPU thread).
# operator rules (2026-10-21): a reactive voxel run BYPASSES the heures creuses rule (any hour, one run per release);
# the series scene is Mont-Saint-Michel (prompts/msm-v1.txt); only a gate PASS becomes a video post, a fail is one line
# in the later full board; framing is "first look", quant + ctx + 5090 on screen; speed claims stay llama.cpp.
# run AFTER day0-fastlane says the model loads on a llama.cpp build. same harness as 66v/66w (voxelbench/).
#
# usage (env): MODEL_FILE=<gguf> SLUG=<slug> [ALIAS=<served id>] [BIN=<llama-server>] [CTX=131072]
#              [KV=q8_0] [EXTRA_FLAGS="--spec-type draft-mtp --spec-draft-n-max 2"] [WALL=60] [TURNS=20]
#              PARAMS_SRC="<card url> fetched <date>" [SAMPLING_FLAGS="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0"] [RB=32768] [PROMPT=<prompt file>]
# release-day rule (2026-10-22, Flint first pass: no sampling flags + our Qwen 16k budget, weak scene): pass the MODEL CARD's
# recommended sampling as SAMPLING_FLAGS and its output budget as RB (RB >= the card's budget incl. reasoning). without flags the
# server uses the GGUF's general.sampling.* keys when present (llama.cpp applies them as defaults; Flint's carry temp/top-k/top-p
# but no min_p), else llama.cpp defaults (temp 0.8, top-k 40, top-p 0.95, min-p 0.05). the /props line below shows what applied.
# Hermes' main-loop requests carry no temperature/top_p/top_k/min_p/max_tokens (CPU smoke 2026-10-22), so server flags reach them.
# launch: tmux new-session -d -s vx-<slug> 'DRYRUN=1 MODEL_FILE=.. SLUG=.. bash ~/day0-voxel.sh >>~/day0-voxel-<slug>.log 2>&1 && MODEL_FILE=.. SLUG=.. bash ~/day0-voxel.sh >>~/day0-voxel-<slug>.log 2>&1'
# artifact: ~/voxelbench/runs/<slug>/gate/gate.json (+ clip, driver.json, metrics.json, server-metrics.txt)
set -uo pipefail
: "${MODEL_FILE:?MODEL_FILE required}"; : "${SLUG:?SLUG required}"
NL_TAG=vx0
MANIFEST=/home/witcheer/day0-voxel-$SLUG-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
BIN=${BIN:-/home/witcheer/llama.cpp/build/bin/llama-server}
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RUNLOG=/home/witcheer/day0-voxel-$SLUG-run.log
ALIAS=${ALIAS:-$(basename "$MODEL_FILE")}
CTX=${CTX:-131072}; KV=${KV:-q8_0}; EXTRA_FLAGS=${EXTRA_FLAGS:-}; SAMPLING_FLAGS=${SAMPLING_FLAGS:-}
WALL=${WALL:-60}; TURNS=${TURNS:-20}
# thinking cap per request (2026-10-21 pilot: one 79k-token thinking block before the first tool call, 71% of ctx):
# same 16k as the board's think-on budget; RB=-1 disables, for non-thinking models it is a no-op
RB=${RB:-16384}
RB_MSG=${RB_MSG:-"Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."}
PROMPT=${PROMPT:-$V/prompts/msm-v1.txt}   # env override (2026-10-22: series moved to msm-v2)
ART=$V/runs/$SLUG/gate/gate.json
# HC bypass by operator rule: the hard stop is our own wall budget, not the tariff window
export NL_HARD_STOP_EPOCH=$(( $(date +%s) + (WALL + 25) * 60 ))

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) slug=$SLUG model=$MODEL_FILE ctx=$CTX kv=$KV rb=$RB extra='$EXTRA_FLAGS' sampling='$SAMPLING_FLAGS' prompt=$(basename "$PROMPT")"
# parameters are identified BEFORE the run (operator 2026-10-22: "identify the 'right parameters' before"): run
#   scripts/card-params.py <hf repo> --gguf <file>, write the values + card url + fetch date into voxelbench/model-params.md,
#   then pass PARAMS_SRC="<card url> fetched <date>" with SAMPLING_FLAGS / RB / EXTRA_FLAGS. refuse otherwise.
PARAMS_SRC=${PARAMS_SRC:-}
[ -n "$PARAMS_SRC" ] || nl_fail "PARAMS_SRC empty: identify the maker parameters first (scripts/card-params.py, voxelbench/model-params.md)"
GG_SAMP=$($GPY - "$MODEL_FILE" <<'PYEOF' 2>/dev/null
import sys
try:
    from gguf import GGUFReader
except ImportError:
    sys.path.insert(0, "/home/witcheer/llama.cpp-donald/gguf-py"); from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); out = []
for k, f in r.fields.items():
    if k.startswith("general.sampling."):
        try: v = f.parts[f.data[0]].tolist()
        except Exception: v = "?"
        out.append(k.split(".")[-1] + "=" + str(v[0] if isinstance(v, list) and v else v))
print(" ".join(out))
PYEOF
)
nl_say "  params: src='$PARAMS_SRC' gguf_sampling='${GG_SAMP:-none}' flags='$SAMPLING_FLAGS'"
[ -n "$SAMPLING_FLAGS" ] || [ -n "$GG_SAMP" ] || nl_fail "the GGUF carries no general.sampling keys and SAMPLING_FLAGS is empty: the run would get llama.cpp defaults (temp 0.8, top-k 40), not the maker's settings"
nl_check_path "$BIN" "$GPY" "$MODEL_FILE" "$PROMPT" "$V/scripts/run_agent.sh" "$V/scripts/gate.py" "$V/scripts/metrics.py" \
  "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
VER=$("$BIN" --version 2>&1 | grep -m1 -i 'version' | tr -d '"'); nl_ok "llama.cpp ${VER:-version unreadable}"
HELP=$("$BIN" --help 2>&1 || true)
case "$HELP" in *--reasoning-budget-message*) nl_ok "--reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build (set BIN to a newer llama.cpp)";; esac
for f in $EXTRA_FLAGS $SAMPLING_FLAGS; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac;; esac; done
nl_check_disk 20
if nl_skip_exact "$ART"; then nl_say "nothing to run, $SLUG already has a gate verdict"; exit 0; fi
# the card must be free: an idle-queue item or any other GPU process means stop and ask, never kill it
if tmux has-session -t =idleq 2>/dev/null; then nl_fail "idleq owns the card ($(cat /home/witcheer/idleq-active.json 2>/dev/null | head -c 160)); pause or wait, then relaunch"; fi
OTHERS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server || true)
[ -z "$OTHERS" ] || nl_fail "other GPU processes: $OTHERS"
nl_say "  planned RUN: $SLUG (wall $WALL min, $TURNS goal turns, + ~3 min gate; HC bypass, own hard stop $(date -d "@$NL_HARD_STOP_EPOCH" '+%H:%M') local)"
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

restore() { tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; nl_donald_restore; sleep 5; curl -s -m5 127.0.0.1:8090/health | grep -qi ok && nl_say "donald health ok" || nl_say "WARN: donald health not ok after restore"; }
trap restore EXIT
nl_say "draining Donald"; nl_donald_stop
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "$(ls -A $V/runs/$SLUG/work 2>/dev/null)" ] && { nl_say "  stale work dir: moving aside"; mv $V/runs/$SLUG $V/runs/$SLUG.stale-$(date +%s); }
mkdir -p $V/runs/$SLUG
# per-run prompt = series prompt + the exact absolute output path (2026-10-21 MSM run saved its deliverable to a
# retyped, misspelled sibling dir runs/qwen38-27b-q6kmsm1/work and the gate saw no html)
PROMPT_RUN=$V/runs/$SLUG/prompt.txt
{ cat "$PROMPT"; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$V/runs/$SLUG/work" "$V/runs/$SLUG/work"; } > "$PROMPT_RUN"

# fit ladder: the requested ctx first, then smaller, never under the 64K Hermes Agent minimum
t0=$(date +%s); nl_mark "$SLUG" running; UP=0
for c in $CTX 98304 65536; do
  [ "$c" -le "$CTX" ] || continue
  # shellcheck disable=SC2086
  $BIN --model "$MODEL_FILE" --host 127.0.0.1 --port 8090 --n-gpu-layers 99 --ctx-size $c --parallel 1 --flash-attn on \
    --cache-type-k $KV --cache-type-v $KV --alias "$ALIAS" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $SAMPLING_FLAGS $EXTRA_FLAGS > $V/runs-$SLUG-server.log 2>&1 &
  for i in $(seq 1 150); do curl -s 127.0.0.1:8090/health | grep -qi ok && { UP=1; break; }; kill -0 $! 2>/dev/null || break; sleep 2; done
  [ "$UP" = 1 ] && { CTX=$c; break; }
  nl_say "  server did not come up at ctx $c: $(grep -m1 -iE 'out of memory|failed|error|unknown' $V/runs-$SLUG-server.log | head -c 200)"
  pkill -9 -x llama-server 2>/dev/null; sleep 4
done
[ "$UP" = 1 ] || { nl_mark "$SLUG" failed "server did not come up at any ctx (see runs-$SLUG-server.log)"; exit 1; }
nl_say "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX"
curl -s 127.0.0.1:8090/props > $V/runs/$SLUG/props.json 2>/dev/null
nl_say "  server sampling defaults: $(python3 -c "import json;g=json.load(open('$V/runs/$SLUG/props.json')).get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')})" 2>/dev/null)"
echo "{\"model_file\": \"$MODEL_FILE\", \"ctx\": $CTX, \"kv\": \"$KV\", \"extra\": \"$EXTRA_FLAGS\", \"sampling\": \"$SAMPLING_FLAGS\", \"gguf_sampling\": \"$GG_SAMP\", \"params_src\": \"$PARAMS_SRC\", \"reasoning_budget\": $RB, \"prompt\": \"$(basename "$PROMPT")\", \"build\": \"$VER\"}" > $V/runs/$SLUG/run-config.json
export DEADLINE_EPOCH=$(( NL_HARD_STOP_EPOCH - 300 ))
# Hermes /goal quality gate (QGATE=0 disables; 2026-10-22 after Muse 66y was judged done on a page that threw a SyntaxError):
# gate.py must pass at a turn boundary before the judge may call the goal done; its output tail goes back to the agent
export VX_TMPFIX=${VX_TMPFIX:-1}   # socket-safe TMPDIR for the model's terminal (HARDENING L9)
[ "${QGATE:-1}" = "1" ] && export GATE_CMD="$GPY $V/scripts/gate.py $V/runs/$SLUG/work/index.html $V/runs/$SLUG/qgate --video-seconds 0"
bash $V/scripts/run_agent.sh "$SLUG" "$ALIAS" $CTX $WALL $TURNS $PROMPT_RUN >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $V/runs/$SLUG/driver.json 2>/dev/null)"
curl -s 127.0.0.1:8090/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > $V/runs/$SLUG/server-metrics.txt 2>/dev/null
pkill -9 -x llama-server 2>/dev/null; sleep 3
HTML=$V/runs/$SLUG/work/index.html
[ -f "$HTML" ] || HTML=$(ls -t $V/runs/$SLUG/work/*.html 2>/dev/null | head -1)
# path-miss fallback: one run owns the card at a time, so a fresh html anywhere under runs/ is this run's output.
# graded, but flagged in path-miss.json and the log (a path miss is reported, never hidden)
if [ -z "$HTML" ] || [ ! -f "$HTML" ]; then
  STRAY=$(find $V/runs -name '*.html' -newermt "@$t0" -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$STRAY" ]; then HTML=$STRAY; echo "{\"path_miss\": true, \"found\": \"$STRAY\"}" > $V/runs/$SLUG/path-miss.json; nl_say "  PATH MISS: no html in work/, grading the fresh file at $STRAY (flagged)"; fi
fi
mkdir -p $V/runs/$SLUG/gate
if [ -n "$HTML" ] && [ -f "$HTML" ]; then
  setsid timeout 300 $GPY $V/scripts/gate.py "$HTML" $V/runs/$SLUG/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$V/runs/$SLUG/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $V/runs/$SLUG/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc, ctx $CTX"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('PASS' if d.get('pass') else 'FAIL (no video post)', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'))")"
fi
nl_say "DONE $SLUG"
exit 0
