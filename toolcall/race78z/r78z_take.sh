#!/bin/bash
# r78z_take.sh: ONE filmed take of the item-78z race: one real Hermes Agent session against an ALREADY RUNNING OpenAI-compatible server (vLLM or llama-server), behind the counting proxy.
# usage: r78z_take.sh <takedir>      env: UPSTREAM (http://127.0.0.1:PORT, required)  ALIAS (model id, required)  TAG (nvfp4|q4km, required)  PROMPT (file)  WALL_S (720)
#        CTXCFG (65536)  FR_PORT (proxy port, 8291)  THINK (on)  FR_PY (python with pyte)  FR_HBIN (Hermes launcher)  FR_CHECKOUT  FR_QUIET (20)  ARENA (/tmp/arena78z)  COLS (80) ROWS (24)
# what it does: counting proxy -> a private HERMES_HOME (unique path per take) -> one tmux session r78z-<tag> running r78z_record.py -> typed, ready -> GO at once (one agent: the
#   barrier is trivial; the two takes are START-SYNCED later, by the renderer, on each take's own Enter mark) -> wait for done/wall -> copy the work dir + state.db out -> stop ONLY what
#   it started (tmux session, proxy, the home's own idle-exit gateway via `gateway stop` scoped by HERMES_HOME). Never starts or stops a model server, never touches Donald, never kills by name.
# exit 0 = the agent ended (done or wall), 1 = setup failure.
set -uo pipefail
RUN=${1:?takedir}
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=${UPSTREAM:?}; ALIAS=${ALIAS:?}; TAG=${TAG:?}
PROMPT=${PROMPT:-$HERE/prompts/voxel-cottage-v1.txt}
WALL_S=${WALL_S:-720}; CTXCFG=${CTXCFG:-65536}; FR_PORT=${FR_PORT:-8291}; FR_QUIET=${FR_QUIET:-20}; THINK=${THINK:-on}
FR_PY=${FR_PY:-/home/witcheer/scratch/race/venv/bin/python}
FR_CHECKOUT=${FR_CHECKOUT:-/home/witcheer/.hermes/hermes-agent}
FR_HBIN=${FR_HBIN:-$FR_CHECKOUT/.hermes/bin/hermes}
ARENA=${ARENA:-/tmp/arena78z}
COLS=${COLS:-76}; ROWS=${ROWS:-22}
SESS=r78z-$TAG
say() { echo "[r78z-take $TAG] $(date +%H:%M:%S) $*"; }
mkdir -p "$RUN"/{rec,hud,work,final,state}
PROXY_PID=""
stop_gateway() { [ -n "${HOMEROOT:-}" ] && [ -d "$HOMEROOT/home-$TAG" ] && HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME="$HOMEROOT/home-$TAG" timeout 60 "$FR_HBIN" gateway stop >/dev/null 2>&1; }
cleanup() { tmux kill-session -t "=$SESS" 2>/dev/null; [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""; stop_gateway; }
trap cleanup EXIT
for f in "$PROMPT" "$HERE/r78z_proxy.py" "$HERE/r78z_record.py" "$HERE/r78z_mkhome.py" "$FR_PY" "$FR_HBIN"; do [ -e "$f" ] || { say "FAIL missing $f"; exit 1; }; done
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$UPSTREAM/health"); [ "$code" = 200 ] || { say "FAIL upstream $UPSTREAM /health answered '$code'"; exit 1; }
ss -ltnH "sport = :$FR_PORT" | grep -q . && { say "FAIL proxy port $FR_PORT is taken"; exit 1; }
tmux has-session -t "=$SESS" 2>/dev/null && { say "FAIL tmux session $SESS already exists"; exit 1; }
mkdir -p "$ARENA"
TS=$(date +%s)
# a per-take home root: an auto-started per-home gateway is keyed by the home PATH and idles out 10 min after its client leaves (measured 2026-11-03, item 80)
HOMEROOT=$ARENA/r$TS-$TAG
mkdir -p "$HOMEROOT"
WORK=$ARENA/$TAG; [ -e "$WORK" ] && mv "$WORK" "$WORK.old-$TS"; mkdir -p "$WORK"
python3 "$HERE/r78z_proxy.py" "$UPSTREAM" "$FR_PORT" "$RUN/hud" --agents a > "$RUN/hud/proxy.log" 2>&1 &
PROXY_PID=$!
for i in $(seq 1 20); do grep -q "four_proxy up" "$RUN/hud/proxy.log" 2>/dev/null && break; sleep 0.5; done
grep -q "four_proxy up" "$RUN/hud/proxy.log" || { say "FAIL proxy did not start: $(head -c 300 "$RUN/hud/proxy.log")"; exit 1; }
say "proxy up (pid $PROXY_PID) :$FR_PORT -> $UPSTREAM"
python3 "$HERE/r78z_mkhome.py" "$HOMEROOT" "$TAG" "$FR_PORT" "$ALIAS" "$CTXCFG" --think "$THINK" > /dev/null || { say "FAIL mkhome"; exit 1; }
sed "s#__WORKDIR__#$WORK#g" "$PROMPT" > "$RUN/rec/prompt.txt"
rm -f "$RUN/rec/ready" "$RUN/rec/result.json" "$RUN/rec/GO" "$RUN/rec/rc"
tmux new-session -d -s "$SESS" -x "$COLS" -y "$ROWS" \
  "$FR_PY $HERE/r78z_record.py --agent a --out $RUN/rec/events-a.jsonl --prompt $RUN/rec/prompt.txt --workdir $WORK --home $HOMEROOT/hm --hermes-home $HOMEROOT/home-$TAG --hermes-bin $FR_HBIN --cols $COLS --rows $ROWS --wall-s $WALL_S --go $RUN/rec/GO --ready $RUN/rec/ready --state-url http://127.0.0.1:$FR_PORT/_state --quiet-s $FR_QUIET --final $RUN/final/pane-a.txt --result $RUN/rec/result.json 2> $RUN/rec/rec.err; echo rc=\$? > $RUN/rec/rc; sleep 3"
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
cp -a "$HOMEROOT/home-$TAG/state.db" "$RUN/state/" 2>/dev/null
say "work dir + state.db copied; stopping the proxy and the session"
cleanup
exit 0
