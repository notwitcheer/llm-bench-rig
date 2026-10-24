#!/bin/bash
# hosted-voxel.sh: voxel series run for a HOSTED model through Nous Portal (2026-10-24, mercury, GPU thread).
# same harness as day0-voxel.sh (run_agent.sh, msm-v2 + exact path line, /goal quality gate, gate.py on the 5090, metrics),
# but inference is remote, so: no llama-server, no Donald drain, no fit ladder, and it NEVER kills a llama-server
# (an idle-queue bench may own one). the card is touched only by the gate + clip (~1-3 min, can share with a queue item;
# fps is then not a claim). a local relay (scripts/portal_relay.py) holds the credential, logs usage, enforces a hard
# spend ceiling and is the pid run_agent.sh SIGSTOPs during gate setup.
#
# usage (env): MODEL_ID=<portal slug> SLUG=<slug> PARAMS_SRC="<maker doc url> fetched <date>" KEY_SOURCE=<hermes_home:DIR|file:PATH>
#              RATES=<in,cache_read,out USD per M> CEIL=<USD> [EFFORT=omit|low|high|max|passthrough] [CTX=131072]
#              [WALL=60] [TURNS=20] [PROMPT=<prompt file>] [RELAY_PORT=8191] [QGATE=1]
# launch: tmux new-session -d -s vx-<slug> 'DRYRUN=1 ... bash ~/voxelbench/scripts/hosted-voxel.sh >>~/hosted-voxel-<slug>.log 2>&1 && ... bash ~/voxelbench/scripts/hosted-voxel.sh >>~/hosted-voxel-<slug>.log 2>&1'
# artifact: ~/voxelbench/runs/<slug>/gate/gate.json (+ relay-log.jsonl, relay-state.json, driver.json, metrics.json)
set -uo pipefail
: "${MODEL_ID:?MODEL_ID required}"; : "${SLUG:?SLUG required}"
NL_TAG=vxh
MANIFEST=/home/witcheer/hosted-voxel-$SLUG-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
HPY=/home/witcheer/.hermes/hermes-agent/venv/bin/python
RUNLOG=/home/witcheer/hosted-voxel-$SLUG-run.log
CTX=${CTX:-131072}; WALL=${WALL:-60}; TURNS=${TURNS:-20}; EFFORT=${EFFORT:-passthrough}
RELAY_PORT=${RELAY_PORT:-8191}
UPSTREAM=${UPSTREAM:-https://inference-api.nousresearch.com/v1}   # override only for the CPU tiny-model harness test
PROMPT=${PROMPT:-$V/prompts/msm-v2.txt}
ART=$V/runs/$SLUG/gate/gate.json
R=$V/runs/$SLUG
export NL_HARD_STOP_EPOCH=$(( $(date +%s) + (WALL + 25) * 60 ))

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) slug=$SLUG model=$MODEL_ID ctx=$CTX effort=$EFFORT wall=$WALL turns=$TURNS prompt=$(basename "$PROMPT") ceiling=${CEIL:-unset} USD"
[ -n "${PARAMS_SRC:-}" ] || nl_fail "PARAMS_SRC empty: identify the maker parameters first (voxelbench/model-params.md)"
[ -n "${KEY_SOURCE:-}" ] || nl_fail "KEY_SOURCE empty (hermes_home:<dir> or file:<path>)"
[ -n "${RATES:-}" ] || nl_fail "RATES empty (USD per M: input,cache_read,output from the live catalogue)"
[ -n "${CEIL:-}" ] || nl_fail "CEIL empty (USD spend ceiling)"
nl_check_path "$GPY" "$HPY" "$PROMPT" "$V/scripts/run_agent.sh" "$V/scripts/gate.py" "$V/scripts/metrics.py" "$V/scripts/portal_relay.py" \
  "$V/lib/libasound.so.2" "$V/ffmpeg" /home/witcheer/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
nl_check_disk 20
ss -ltnH "sport = :$RELAY_PORT" 2>/dev/null | grep -q . && nl_fail "port $RELAY_PORT already in use"
# credential check without printing anything: the relay's own resolver, key length only
case "${KEY_SOURCE:-}" in
  hermes_home:*) D=${KEY_SOURCE#hermes_home:}; KL=$(cd /home/witcheer/.hermes/hermes-agent && HERMES_HOME=$D PYTHONPATH=/home/witcheer/.hermes/hermes-agent $HPY -c "from hermes_cli.auth_nous import resolve_nous_runtime_credentials as r; d=r(); print(len(d['api_key'] or ''), int(d.get('expires_in') or 0))" 2>/dev/null); [ -n "$KL" ] && nl_ok "credential resolves (key len, ttl s: $KL)" || nl_fail "credential did not resolve from $D";;
  file:*) F=${KEY_SOURCE#file:}; [ -s "$F" ] && nl_ok "key file present ($(stat -c %a "$F") perms)" || nl_fail "key file $F missing/empty";;
esac
if nl_skip_exact "$ART"; then nl_say "nothing to run, $SLUG already has a gate verdict"; exit 0; fi
tmux has-session -t =idleq 2>/dev/null && nl_say "  note: idleq holds the card; the gate shares it (fps not a claim). nothing of the queue is touched"
nl_say "  planned RUN: $SLUG (wall $WALL min, $TURNS goal turns, + ~3 min gate; own hard stop $(date -d "@$NL_HARD_STOP_EPOCH" '+%H:%M') local)"
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$SLUG"; fi
nl_gate

[ "$(ls -A $R/work 2>/dev/null)" ] && { nl_say "  stale work dir: moving aside"; mv $R $R.stale-$(date +%s); }
mkdir -p $R
PROMPT_RUN=$R/prompt.txt
{ cat "$PROMPT"; printf '\nThe working directory is %s. The final file must be exactly %s/index.html (copy this path, never retype it).\n' "$R/work" "$R/work"; } > "$PROMPT_RUN"

RPID=""
cleanup() { tmux kill-session -t "=voxel-$SLUG" 2>/dev/null; [ -n "$RPID" ] && kill "$RPID" 2>/dev/null; nl_say "EXIT trap: relay stopped (pid ${RPID:-none}); no llama-server or Donald touched"; }
trap cleanup EXIT
t0=$(date +%s); nl_mark "$SLUG" running
$HPY $V/scripts/portal_relay.py --port $RELAY_PORT --upstream "$UPSTREAM" --key-source "$KEY_SOURCE" --hermes-python $HPY --rates "$RATES" \
  --ceiling-usd "$CEIL" --reasoning-effort "$EFFORT" --include-usage --log $R/relay-log.jsonl --state $R/relay-state.json > $R/relay.out 2>&1 &
RPID=$!
for i in $(seq 1 30); do curl -s -m2 127.0.0.1:$RELAY_PORT/health | grep -q '"ok"' && break; kill -0 $RPID 2>/dev/null || break; sleep 1; done
curl -s -m2 127.0.0.1:$RELAY_PORT/health | grep -q '"ok"' || { nl_mark "$SLUG" failed "relay did not come up: $(tail -c 300 $R/relay.out)"; exit 1; }
nl_say "relay up pid $RPID: $(head -1 $R/relay.out)"
echo "{\"model_id\": \"$MODEL_ID\", \"route\": \"nous portal via local relay\", \"ctx\": $CTX, \"effort\": \"$EFFORT\", \"rates_per_m\": \"$RATES\", \"ceiling_usd\": $CEIL, \"params_src\": \"$PARAMS_SRC\", \"prompt\": \"$(basename "$PROMPT")\", \"hermes\": \"$(/home/witcheer/.local/bin/hermes --version 2>&1 | head -1)\"}" > $R/run-config.json
export DEADLINE_EPOCH=$(( NL_HARD_STOP_EPOCH - 300 ))
export VX_TMPFIX=${VX_TMPFIX:-1}
export PORT=$RELAY_PORT    # run_agent.sh points the profile at the relay and SIGSTOPs the pid on this port for the gate setup
[ "${QGATE:-1}" = "1" ] && export GATE_CMD="$GPY $V/scripts/gate.py $R/work/index.html $R/qgate --video-seconds 0"
bash $V/scripts/run_agent.sh "$SLUG" "$MODEL_ID" $CTX $WALL $TURNS $PROMPT_RUN >> "$RUNLOG" 2>&1
nl_say "driver: $(cat $R/driver.json 2>/dev/null)"
kill $RPID 2>/dev/null; RPID=""
nl_say "relay totals: $(cat $R/relay-state.json 2>/dev/null)"
HTML=$R/work/index.html
[ -f "$HTML" ] || HTML=$(ls -t $R/work/*.html 2>/dev/null | head -1)
if [ -z "$HTML" ] || [ ! -f "$HTML" ]; then
  # path-miss fallback scoped to THIS run dir (other runs may write under runs/ on a shared day)
  STRAY=$(find $R -name '*.html' -newermt "@$t0" -size +1k -not -path "*/gate/*" -not -path "*/qgate/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$STRAY" ]; then HTML=$STRAY; echo "{\"path_miss\": true, \"found\": \"$STRAY\"}" > $R/path-miss.json; nl_say "  PATH MISS: grading $STRAY (flagged)"; fi
fi
mkdir -p $R/gate
if [ -n "$HTML" ] && [ -f "$HTML" ]; then
  setsid timeout 300 $GPY $V/scripts/gate.py "$HTML" $R/gate --video-seconds 20 < /dev/null >> "$RUNLOG" 2>&1; grc=$?
else
  echo '{"pass": false, "no_html": true}' > "$ART"; grc=1
fi
python3 $V/scripts/metrics.py /home/witcheer/.hermes/profiles/$(python3 -c "import json;print(json.load(open('$R/driver.json'))['profile'])" 2>/dev/null || echo vx${SLUG//[^a-z0-9]/}) $R/metrics.json >> "$RUNLOG" 2>&1
if nl_judge_artifact "$SLUG" 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min, gate rc $grc"; then
  nl_say "  gate: $(python3 -c "import json;d=json.load(open('$ART'));print('PASS' if d.get('pass') else 'FAIL (no video post)', d.get('checks'), 'fps', d.get('fps'), 'bytes', d.get('bytes'))")"
fi
nl_say "DONE $SLUG"
exit 0
