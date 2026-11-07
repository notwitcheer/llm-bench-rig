#!/bin/bash
# wd_take.sh: the ONE filmed editor take of item 95: a real Hermes Agent session (the editor) driving the sandbox tool `wd` against an ALREADY RUNNING llama-server (the editor model), behind the counting proxy.
# usage: wd_take.sh <takedir>      env: UPSTREAM (http://127.0.0.1:PORT of the editor model, required)  ALIAS (model id, required)  PROMPT (file)  WALL_S (cap, s)  CTXCFG (65536)  FR_PORT (proxy port)
#        SVC_PORT (tool service port)  SVC_STUB=1 (service without a GPU: CPU proof)  EVALS  MINUTES  DEV_LIMIT  EVAL_CAP  EVAL_PORT  BASE_ROWS (dev rows of the base model)  TABLES (loop/prod activation tables)
#        FR_PY (python with pyte)  FR_HBIN (Hermes launcher)  FR_CHECKOUT  FR_QUIET (20)  ARENA (/tmp/arena95)  COLS ROWS  CODE (harness dir)
# what it does: counting proxy -> tool service -> a private HERMES_HOME (unique path per take, never the main gateway) -> one tmux session running wd_record.py -> typed, ready, GO -> wait for done/wall -> copy the work dir, the service state and state.db out
#   -> stop ONLY what it started (tmux session, proxy, service, the home's own idle-exit gateway via `gateway stop` scoped by HERMES_HOME). Never starts or stops a model server, never touches Donald, never kills by name.
# exit 0 = the agent ended (done or wall), 1 = setup failure.
set -uo pipefail
RUN=${1:?takedir}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=${CODE:-$HERE}
UPSTREAM=${UPSTREAM:?}; ALIAS=${ALIAS:?}
PROMPT=${PROMPT:-$HERE/prompts/edit-task.txt}
WALL_S=${WALL_S:-3600}; CTXCFG=${CTXCFG:-65536}; FR_PORT=${FR_PORT:-8295}; SVC_PORT=${SVC_PORT:-8285}; FR_QUIET=${FR_QUIET:-25}
EVALS=${EVALS:-6}; MINUTES=${MINUTES:-55}; DEV_LIMIT=${DEV_LIMIT:-0}; EVAL_CAP=${EVAL_CAP:-16384}; EVAL_PORT=${EVAL_PORT:-8283}; EVAL_SLOTS=${EVAL_SLOTS:-20}
FR_PY=${FR_PY:-/home/witcheer/scratch/race/venv/bin/python}
SPY=${SPY:-/home/witcheer/benchmark-rig/venv/bin/python}      # python for the tool service (numpy, datasets, transformers)
FR_CHECKOUT=${FR_CHECKOUT:-/home/witcheer/.hermes/hermes-agent}
FR_HBIN=${FR_HBIN:-$FR_CHECKOUT/.hermes/bin/hermes}
ARENA=${ARENA:-/tmp/arena95}
COLS=${COLS:-66}; ROWS=${ROWS:-30}
SESS=wd-editor
say() { echo "[wd-take] $(date +%H:%M:%S) $*"; }
mkdir -p "$RUN"/{rec,hud,work,final,state}
PROXY_PID=""; PW_PID=""; SVC_PID=""
stop_gateway() { [ -n "${HOMEROOT:-}" ] && [ -d "$HOMEROOT/home-ed" ] && HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME="$HOMEROOT/home-ed" timeout 60 "$FR_HBIN" gateway stop >/dev/null 2>&1; }
cleanup() { tmux kill-session -t "=$SESS" 2>/dev/null; [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""; [ -n "$PW_PID" ] && kill "$PW_PID" 2>/dev/null; PW_PID=""
  [ -n "$SVC_PID" ] && { kill "$SVC_PID" 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$SVC_PID" 2>/dev/null || break; sleep 1; done; kill -9 "$SVC_PID" 2>/dev/null; }; SVC_PID=""; stop_gateway; }
trap cleanup EXIT
for f in "$PROMPT" "$CODE/wd_proxy.py" "$CODE/wd_record.py" "$CODE/wd_mkhome.py" "$CODE/wd_service.py" "$CODE/wd_cli.py" "$FR_PY" "$SPY" "$FR_HBIN" "$BASE_ROWS" "$TABLES/loop_mean.npy"; do [ -e "$f" ] || { say "FAIL missing $f"; exit 1; }; done
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$UPSTREAM/health"); [ "$code" = 200 ] || { say "FAIL upstream $UPSTREAM /health answered '$code'"; exit 1; }
for p in $FR_PORT $SVC_PORT; do ss -ltnH "sport = :$p" | grep -q . && { say "FAIL port $p is taken"; exit 1; }; done
tmux has-session -t "=$SESS" 2>/dev/null && { say "FAIL tmux session $SESS already exists"; exit 1; }
mkdir -p "$ARENA"
TS=$(date +%s)
HOMEROOT=$ARENA/r$TS
mkdir -p "$HOMEROOT"
WORK=$ARENA/work-$TS; mkdir -p "$WORK"
python3 "$CODE/wd_proxy.py" "$UPSTREAM" "$FR_PORT" "$RUN/hud" --agents a > "$RUN/hud/proxy.log" 2>&1 &
PROXY_PID=$!
for i in $(seq 1 20); do grep -q "four_proxy up" "$RUN/hud/proxy.log" 2>/dev/null && break; sleep 0.5; done
grep -q "four_proxy up" "$RUN/hud/proxy.log" || { say "FAIL proxy did not start: $(head -c 300 "$RUN/hud/proxy.log")"; exit 1; }
say "proxy up (pid $PROXY_PID) :$FR_PORT -> $UPSTREAM"
( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')" >> "$RUN/hud/power.csv"; sleep 1; done ) >/dev/null 2>&1 &
PW_PID=$!
# the sandbox tool service (own port, own state dir; held-out problems are not reachable from it)
STUBF=""; [ "${SVC_STUB:-0}" = 1 ] && STUBF="--stub"
LIMF=""; [ "$DEV_LIMIT" != 0 ] && LIMF="--limit $DEV_LIMIT"
"$SPY" "$CODE/wd_service.py" --port "$SVC_PORT" --state "$RUN/state" --work "$WORK" --base-rows "$BASE_ROWS" --tables "$TABLES" --evals "$EVALS" --minutes "$MINUTES" --eval-port "$EVAL_PORT" --cap "$EVAL_CAP" --slots "$EVAL_SLOTS" --acts-device "${ACTS_DEVICE:-cpu}" $LIMF $STUBF > "$RUN/state/service.log" 2>&1 &
SVC_PID=$!
for i in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$SVC_PORT/health" | grep -q ok && break; kill -0 "$SVC_PID" 2>/dev/null || break; sleep 1; done
curl -s -m 2 "http://127.0.0.1:$SVC_PORT/health" | grep -q ok || { say "FAIL tool service did not start: $(tail -c 300 "$RUN/state/service.log")"; exit 1; }
say "tool service up (pid $SVC_PID) :$SVC_PORT, work $WORK, evals $EVALS, minutes $MINUTES"
# the editor's `wd` command: a wrapper at a neutral path (the screen shows this path, not the harness dir)
TOOLS=$ARENA/tools-$TS; mkdir -p "$TOOLS"
cp "$CODE/wd_cli.py" "$TOOLS/wd_cli.py"
printf '#!/bin/sh\nWD_PORT=%s exec /usr/bin/python3 %s/wd_cli.py "$@"\n' "$SVC_PORT" "$TOOLS" > "$TOOLS/wd"; chmod 755 "$TOOLS/wd"
# the editor's own `python3` = a venv with numpy only (a neutral path; no torch, no model files, no network libraries beyond the stdlib)
NPY=${NPY:-/home/witcheer/voxel-stage/wedit95/npvenv/bin/python3}; [ -x "$NPY" ] || { say "FAIL numpy venv missing: $NPY"; exit 1; }
printf '#!/bin/sh\nexec %s "$@"\n' "$NPY" > "$TOOLS/python3"; chmod 755 "$TOOLS/python3"; ln -sf python3 "$TOOLS/python"
NPSITE=$(ls -d "$(dirname "$(dirname "$NPY")")"/lib/python3*/site-packages | head -1)
python3 "$CODE/wd_mkhome.py" "$HOMEROOT" ed "$FR_PORT" "$ALIAS" "$CTXCFG" > /dev/null || { say "FAIL mkhome"; exit 1; }
sed -e "s#__WORKDIR__#$WORK#g" -e "s#__PY__#$TOOLS/python3#g" "$PROMPT" > "$RUN/rec/prompt.txt"
grep -q "__WORKDIR__\|__PY__" "$RUN/rec/prompt.txt" && { say "FAIL prompt still has a placeholder"; exit 1; }
rm -f "$RUN/rec/ready" "$RUN/rec/result.json" "$RUN/rec/GO" "$RUN/rec/rc"
tmux new-session -d -s "$SESS" -x "$COLS" -y "$ROWS" \
  "RP_TOOLS=$TOOLS WD_PYTHONPATH=$NPSITE $FR_PY $CODE/wd_record.py --agent a --out $RUN/rec/events-a.jsonl --prompt $RUN/rec/prompt.txt --workdir $WORK --home $HOMEROOT/hm --hermes-home $HOMEROOT/home-ed --hermes-bin $FR_HBIN --cols $COLS --rows $ROWS --wall-s $WALL_S --max-turns 90 --go $RUN/rec/GO --ready $RUN/rec/ready --state-url http://127.0.0.1:$FR_PORT/_state --quiet-s $FR_QUIET --final $RUN/final/pane-a.txt --result $RUN/rec/result.json 2> $RUN/rec/rec.err; echo rc=\$? > $RUN/rec/rc; sleep 3"
say "recorder launched (tmux $SESS); waiting for the prompt to be typed"
for i in $(seq 1 240); do
  [ -f "$RUN/rec/ready" ] && break
  [ -f "$RUN/rec/rc" ] && { say "FAIL recorder ended before the start: $(cat "$RUN/rec/rc") / $(tail -c 300 "$RUN/rec/rec.err")"; exit 1; }
  sleep 1
done
[ -f "$RUN/rec/ready" ] || { say "FAIL prompt not typed after 240 s"; exit 1; }
sleep 1
T0=$(date +%s.%N); date -u +%FT%TZ > "$RUN/rec/GO"; echo "$T0" > "$RUN/rec/go-epoch.txt"
say "GO: prompt submitted at epoch $T0 (wall cap $WALL_S s)"
END_BY=$(( $(date +%s) + WALL_S + 150 ))
while :; do
  [ -f "$RUN/rec/rc" ] && break
  [ "$(date +%s)" -ge "$END_BY" ] && { say "WARN recorder still running at wall+150 s"; break; }
  sleep 3
done
say "agent: $(cat "$RUN/rec/rc" 2>/dev/null || echo rc=?) $(cat "$RUN/rec/result.json" 2>/dev/null)"
sleep 2
cp -a "$WORK/." "$RUN/work/" 2>/dev/null
python3 -c 'import sqlite3,sys; a=sqlite3.connect("file:%s?mode=ro"%sys.argv[1],uri=True); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close()' "$HOMEROOT/home-ed/state.db" "$RUN/state/agent-state.db" 2>/dev/null || cp -a "$HOMEROOT/home-ed/state.db"* "$RUN/state/" 2>/dev/null
# a dev evaluation still in flight at the wall is waited for (its result may be the best candidate): the service stays up until the GPU job ends or 12 min pass
for i in $(seq 1 72); do
  R=$(curl -s -m 3 "http://127.0.0.1:$SVC_PORT/status" 2>/dev/null)
  case "$R" in *'"running"'*|*'"queued"'*) sleep 10;; *) break;; esac
done
say "work dir + state.db copied; stopping the proxy, the service and the session"
cleanup
exit 0
