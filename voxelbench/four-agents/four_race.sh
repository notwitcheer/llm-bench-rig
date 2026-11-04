#!/bin/bash
# four_race.sh: run the four-agent race against an ALREADY RUNNING OpenAI-compatible server (item 80 helper; also the CPU mock test).
# usage: four_race.sh <rundir>        env: UPSTREAM (http://127.0.0.1:PORT, required)  ALIAS (model id, required)  PROMPT (file)  WALL_S (2400)  CTXCFG (65536)
#        FR_PORT (first proxy port, 8191)  FR_PY (python with pyte, for the recorder)  FR_HBIN (Hermes launcher)  FR_CHECKOUT (Hermes checkout root)  FR_QUIET (20)
#        FR_COMPACT_TOKENS (80b: per-agent compaction budget, written to each home as compression.threshold_tokens; empty = item-80 v1 homes)
#        FR_POOL (80b: KV pool size in tokens, switches the proxy's admission guard on; empty = off)  FR_GUARD_RESERVE_GEN / FR_GUARD_RESERVE_FIRST
#        ARENA (/tmp/arena80: agent work dirs, throwaway HOMEs and HERMES_HOMEs live here, short and free of the account name)
# what it does: counting proxy (one port per agent) -> four private HERMES_HOMEs -> four tmux sessions f80-<a> each running four_record.py -> barrier
#   (all four typed and ready) -> GO -> wait for done / wall -> copy the four work dirs + panes out -> stop ONLY what it started. Exit 0 = all four ended (done or wall),
#   1 = setup failure. It never starts or stops a model server, never touches Donald and never kills by name.
set -uo pipefail
RUN=${1:?rundir}
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=${UPSTREAM:?}; ALIAS=${ALIAS:?}
PROMPT=${PROMPT:-$HERE/prompts/lighthouse-v1.txt}
WALL_S=${WALL_S:-2400}; CTXCFG=${CTXCFG:-65536}; FR_PORT=${FR_PORT:-8191}; FR_QUIET=${FR_QUIET:-20}
FR_PY=${FR_PY:-/home/witcheer/scratch/race/venv/bin/python}
FR_CHECKOUT=${FR_CHECKOUT:-/home/witcheer/.hermes/hermes-agent}
FR_HBIN=${FR_HBIN:-$FR_CHECKOUT/.hermes/bin/hermes}
ARENA=${ARENA:-/tmp/arena80}
COLS=${COLS:-78}; ROWS=${ROWS:-18}
AGENTS=(a b c d)
say() { echo "[four80] $(date +%H:%M:%S) $*"; }
mkdir -p "$RUN"/{rec,hud,work,final}
PROXY_PID=""
stop_gateways() {   # each agent home auto-started its own idle-exit gateway: stop exactly those (HERMES_HOME scopes `gateway stop` to that home; the main gateway is never touched)
  [ -n "${HOMEROOT:-}" ] || return 0
  for a in "${AGENTS[@]}"; do [ -d "$HOMEROOT/home-$a" ] && HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME="$HOMEROOT/home-$a" timeout 60 "$FR_HBIN" gateway stop >/dev/null 2>&1; done
}
cleanup() { for a in "${AGENTS[@]}"; do tmux kill-session -t "=f80-$a" 2>/dev/null; done; [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""; stop_gateways; }
trap cleanup EXIT
for f in "$PROMPT" "$HERE/four_proxy.py" "$HERE/four_record.py" "$HERE/four_mkhome.py" "$FR_PY" "$FR_HBIN"; do [ -e "$f" ] || { say "FAIL missing $f"; exit 1; }; done
curl -s -m 5 "$UPSTREAM/health" | grep -qi ok || { say "FAIL upstream $UPSTREAM not healthy"; exit 1; }
for i in 0 1 2 3; do ss -ltnH "sport = :$((FR_PORT+i))" | grep -q . && { say "FAIL proxy port $((FR_PORT+i)) is taken"; exit 1; }; done
for a in "${AGENTS[@]}"; do tmux has-session -t "=f80-$a" 2>/dev/null && { say "FAIL tmux session f80-$a already exists"; exit 1; }; done
# throwaway agent dirs: wipe only our own previous leftovers under $ARENA/<agent> (never anything else)
mkdir -p "$ARENA"
TS=$(date +%s)
# homes live under a per-run root: an auto-started per-home gateway is keyed by the home PATH and idles out only 10 min after its client leaves, so a reused path would
# collide with the last run's still-live gateway ("a gateway is already running", every pane dead at start, measured 2026-11-03)
HOMEROOT=$ARENA/r$TS
mkdir -p "$HOMEROOT"
for a in "${AGENTS[@]}"; do [ -e "$ARENA/$a" ] && mv "$ARENA/$a" "$ARENA/$a.old-$TS"; mkdir -p "$ARENA/$a"; done
PXARGS=()
[ -n "${FR_POOL:-}" ] && PXARGS=(--pool "$FR_POOL" --reserve-gen "${FR_GUARD_RESERVE_GEN:-12288}" --reserve-first "${FR_GUARD_RESERVE_FIRST:-26624}")
python3 "$HERE/four_proxy.py" "$UPSTREAM" "$FR_PORT" "$RUN/hud" "${PXARGS[@]}" > "$RUN/hud/proxy.log" 2>&1 &
PROXY_PID=$!
for i in $(seq 1 20); do grep -q "four_proxy up" "$RUN/hud/proxy.log" 2>/dev/null && break; sleep 0.5; done
grep -q "four_proxy up" "$RUN/hud/proxy.log" || { say "FAIL proxy did not start: $(head -c 300 "$RUN/hud/proxy.log")"; exit 1; }
say "proxy up (pid $PROXY_PID) ports $FR_PORT..$((FR_PORT+3)) -> $UPSTREAM"
i=0
for a in "${AGENTS[@]}"; do
  port=$((FR_PORT+i)); i=$((i+1))
  FR_COMPACT_TOKENS="${FR_COMPACT_TOKENS:-}" python3 "$HERE/four_mkhome.py" "$HOMEROOT" "$a" "$port" "$ALIAS" "$CTXCFG" > /dev/null || { say "FAIL mkhome $a"; exit 1; }
  sed "s#__WORKDIR__#$ARENA/$a#g" "$PROMPT" > "$RUN/rec/prompt-$a.txt"
  rm -f "$RUN/rec/ready-$a" "$RUN/rec/result-$a.json"
  tmux new-session -d -s "f80-$a" -x "$COLS" -y "$ROWS" \
    "$FR_PY $HERE/four_record.py --agent $a --out $RUN/rec/events-$a.jsonl --prompt $RUN/rec/prompt-$a.txt --workdir $ARENA/$a --home $HOMEROOT/hm-$a --hermes-home $HOMEROOT/home-$a --hermes-bin $FR_HBIN --cols $COLS --rows $ROWS --wall-s $WALL_S --go $RUN/rec/GO --ready $RUN/rec/ready-$a --state-url http://127.0.0.1:$port/_state --quiet-s $FR_QUIET --final $RUN/final/pane-$a.txt --result $RUN/rec/result-$a.json 2> $RUN/rec/rec-$a.err; echo rc=\$? > $RUN/rec/rc-$a; sleep 3"
done
rm -f "$RUN/rec/GO"
say "four recorders launched (tmux f80-a .. f80-d); waiting for all four to be typed and ready"
for i in $(seq 1 240); do
  n=0; for a in "${AGENTS[@]}"; do [ -f "$RUN/rec/ready-$a" ] && n=$((n+1)); done
  [ "$n" = 4 ] && break
  for a in "${AGENTS[@]}"; do [ -f "$RUN/rec/rc-$a" ] && { say "FAIL recorder $a ended before the start: $(cat "$RUN/rec/rc-$a") / $(tail -c 300 "$RUN/rec/rec-$a.err")"; exit 1; }; done
  sleep 1
done
[ "$n" = 4 ] || { say "FAIL only $n of 4 agents ready after 240 s"; exit 1; }
sleep 1
T0=$(date +%s.%N); date -u +%FT%TZ > "$RUN/rec/GO"; echo "$T0" > "$RUN/rec/go-epoch.txt"
say "GO: all four prompts submitted at epoch $T0"
END_BY=$(( $(date +%s) + WALL_S + 150 ))
while :; do
  n=0; for a in "${AGENTS[@]}"; do [ -f "$RUN/rec/rc-$a" ] && n=$((n+1)); done
  [ "$n" = 4 ] && break
  [ "$(date +%s)" -ge "$END_BY" ] && { say "WARN recorders still running at wall+150 s: $n of 4 ended"; break; }
  sleep 5
done
for a in "${AGENTS[@]}"; do say "agent $a: $(cat "$RUN/rec/rc-$a" 2>/dev/null || echo rc=?) $(cat "$RUN/rec/result-$a.json" 2>/dev/null)"; done
sleep 2
for a in "${AGENTS[@]}"; do mkdir -p "$RUN/work/$a"; cp -a "$ARENA/$a/." "$RUN/work/$a/" 2>/dev/null; done
# keep the agents' own session databases next to the recording (messages, tool calls, reasoning chars: the audit trail for 'what did each agent do')
for a in "${AGENTS[@]}"; do mkdir -p "$RUN/state/$a"; cp -a "$HOMEROOT/home-$a"/state.db* "$RUN/state/$a/" 2>/dev/null; cp -a "$HOMEROOT/home-$a/home-version.txt" "$HOMEROOT/home-$a/config.yaml" "$RUN/state/$a/" 2>/dev/null; done
say "work dirs copied to $RUN/work; stopping the proxy and the four sessions"
cleanup
exit 0
