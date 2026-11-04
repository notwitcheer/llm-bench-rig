#!/bin/bash
# chess_agent.sh: ONE Hermes Agent /goal hour for the chess-engine duel (item 81). Derived from voxelbench/scripts/run_agent.sh (same type_line, gate registration order check,
# stall nudge, state.db goal polling) with these changes:
#   - ISOLATED HERMES_HOME: <arena>/home-<agent>, its own data root run by the checkout's minted launcher .hermes/bin/hermes (NOT the checkout venv python: probed 2026-11-03, the venv-python client starts the home's gateway daemon on the 3.14 site-packages and it dies on pydantic_core), never a profile and never the main
#     hermes-gateway.service (the harness stays isolated from the always-on gateway). chess_mkhome.py builds the home.
#   - the model server is on its OWN port ($PORT, default 8581), never :8090; the SIGSTOP gate-setup hold targets only the pid listening on that port.
#   - agent work dir = <arena>/<agent> (short neutral path, no account name in the footage); toolsets terminal + file only.
#   - a pane recorder (pane_rec.py) snapshots the Hermes pane once a second into runs/<slug>/pane-rec/frames.jsonl for the 8x cutaway.
# usage: chess_agent.sh <slug> <agent> <model_id> <ctx> <wall_min> <goal_turns> <prompt_file>      env: PORT ARENA GATE_CMD VX_GATE_STRICT VX_STALL_NUDGE DEADLINE_EPOCH HBIN
set -uo pipefail
SLUG=$1; AGENT=$2; MODEL=$3; CTX=$4; WALL=$5; TURNS=$6; PROMPT_FILE=$7
PORT=${PORT:-8581}; ARENA=${ARENA:-/tmp/chess81}
HERE=$(cd "$(dirname "$0")" && pwd)
V=${V:-/home/witcheer/voxelbench}
B=$V/runs/$SLUG
# the home path is unique per run: re-using a path whose idle-exit gateway daemon may still be alive made the next client exit at startup (probed 2026-11-03)
RUNID=$(date +%s); H=$ARENA/home-$AGENT-$RUNID; W=$ARENA/$AGENT; LOG=$B/driver.log
HCK=${HCK:-/home/witcheer/.hermes/hermes-agent}
HBIN=${HBIN:-$HCK/.hermes/bin/hermes}   # the checkout's minted launcher (store python + dependency generation). `python -m hermes_cli.main` from the checkout venv (python 3.11) starts the home's own gateway daemon with the 3.14 site-packages and dies on pydantic_core (probed 2026-11-03): do not use it
mkdir -p "$B" "$ARENA" "$ARENA/tmp-$AGENT" "$ARENA/hm-$AGENT"
for d in "$W"; do [ -e "$d" ] && mv "$d" "$d.old-$(date +%s)"; done
mkdir -p "$W"
say() { echo "[chess $SLUG] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
python3 "$HERE/chess_mkhome.py" "$ARENA" "$AGENT-$RUNID" "$PORT" "$MODEL" "$CTX" "$TURNS" > /dev/null || { say "mkhome failed"; exit 2; }
[ -f "$H/config.yaml" ] || { say "isolated home not created"; exit 2; }

GOAL=$(tr '\n' ' ' < "$PROMPT_FILE" | sed 's/  */ /g')
S="chess81-$AGENT"
tmux kill-session -t "=$S" 2>/dev/null
# socket-safe temp dir (2026-10-22, HARDENING L9): Hermes points TMPDIR at <profile>/cache/scratch for terminal commands;
# with our long bench profile names that path is ~76 chars and headless Chromium's AF_UNIX socket under it exceeds the
# 108-byte limit, so a model could never launch a browser from its terminal (TargetClosedError; <= 60 chars works).
# Hermes respects a TMPDIR set by the user, so VX_TMPFIX=1 starts Hermes with a short one. default 0 keeps the queued
# one-variable reruns (66z, 66zy) on the same harness as every earlier run; day0-voxel.sh and new items set 1.
# socket-safe short TMPDIR for the model's terminal (HARDENING L9); the arena path is already short
# clean env (a stray HERMES_* or tmux-server variable once pointed a client at someone else's gateway); HOME = a throwaway dir so the agent's ~ is not the account home
tmux new-session -d -s "$S" -x 110 -y 34 "cd $W && env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$ARENA/hm-$AGENT TERM=xterm-256color LANG=C.UTF-8 COLORTERM=truecolor TERMINAL_CWD=$W TMPDIR=$ARENA/tmp-$AGENT HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=$H $HBIN chat --cli --toolsets terminal,file"
mkdir -p "$B/pane-rec"; rm -f "$B/pane-rec/STOP"
python3 "$HERE/pane_rec.py" "$S" "$B/pane-rec/frames.jsonl" "$B/pane-rec/STOP" 1.0 >> "$B/pane-rec/rec.log" 2>&1 &
RECPID=$!
# wait for the prompt to be ready
for i in $(seq 1 60); do sleep 2; tmux capture-pane -t "=$S:" -p | grep -qE "❯|›|> *$" && break; done
sleep 3
tmux capture-pane -t "=$S:" -p > "$B/pane-start.txt"
# type one line into the Hermes TUI and press Enter only once the TUI has taken in all of it (2026-10-28, HARDENING L31).
# Hermes reads an Enter that arrives <50 ms after the last input-buffer change as a pasted newline, not a submit
# (hermes_cli/cli_tui_mixin.py _RAPID_INPUT_ENTER_WINDOW_S, #10994: tmux strips bracketed paste). the old fixed
# "send-keys -l; sleep 0.5; Enter" was fine for the 2.5-3.1 KB msm prompts, but the 6 KB trees-v1 goal was still being
# ingested after 0.5 s: the Enter became a newline, the next typed line (/goal gate add ...) was glued onto the goal
# text, no gate was registered and the model read "/goal gate add" as an instruction (71 trees, 0 goal turns).
# wait until the line's last 32 non-space characters are on screen (whitespace stripped, so wrapping does not matter),
# then 0.6 s more (>> 50 ms), then Enter. fallback after ~30 s: submit anyway and log a WARN.
type_line() {
  local text="$1" tl i
  tl=$(printf '%s' "$text" | tr -d '[:space:]' | tail -c 32)
  tmux send-keys -t "=$S:" -l "$text"
  for i in $(seq 1 100); do
    sleep 0.3
    # no "| grep -q" here: under pipefail, grep -q exiting early SIGPIPEs tr and the pipeline reads as failed
    local scr; scr=$(tmux capture-pane -t "=$S:" -p -S -300 2>/dev/null | tr -d '[:space:]')
    [[ "$scr" == *"$tl"* ]] && break
  done
  if [ "$i" -ge 100 ]; then say "WARN typed line tail never appeared on screen (${#text} chars): submitting anyway"; else say "typed ${#text} chars, on screen after ~$(( i * 3 ))00 ms"; fi
  sleep 0.6
  tmux send-keys -t "=$S:" Enter
}
say "hermes up (isolated home $H, launcher $HBIN), sending /goal (turn budget $TURNS, wall $WALL min, ctx $CTX, model $MODEL, port $PORT)"
T0=$(date +%s)
if [ -n "${GATE_CMD:-}" ]; then
  # Hermes /goal quality gate (docs: user-guide/features/goals#quality-gates). the CLI judges the goal right after
  # each turn and a queued slash command does not defer that judge, so "/goal gate add" typed after "/goal" lands
  # after turn 1's verdict (a one-turn "done" would skip the gate). keep the loop busy with one tiny warm-up turn
  # (system-prompt prefill alone takes seconds), queue /goal then /goal gate add behind it: /goal queues its kickoff
  # at the END of the input queue, so the gate is registered before the first goal turn runs. (2026-10-22)
  # the warm-up must still be running when both lines are typed. a fast model (or a warm prompt cache) answered it in
  # under 2 s in the 19:03 probe and /goal then ran idle -> "/goal gate add: no active goal" after a one-turn "done".
  # deterministic hold: SIGSTOP the model server (idle, waiting on its socket) so the warm-up request hangs, type both
  # lines (they queue), then SIGCONT. the server pauses for ~4 s plus the ingest wait of type_line (a 6 KB goal: a few
  # seconds); a trap guarantees the CONT.
  SPID=$(ss -ltnpH "sport = :${PORT}" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$SPID" ] && { trap 'kill -CONT '"$SPID"' 2>/dev/null' EXIT; kill -STOP "$SPID" 2>/dev/null && say "model server pid $SPID held for the gate setup"; } || say "WARN no server pid on port ${PORT}: gate setup relies on timing"
  type_line "Reply with the single word ready and nothing else."
  sleep 1.5
  type_line "/goal $GOAL"
  sleep 0.8
  type_line "/goal gate add $GATE_CMD"
  sleep 1.2
  [ -n "$SPID" ] && { kill -CONT "$SPID" 2>/dev/null; trap - EXIT; say "model server released"; }
else
  type_line "/goal $GOAL"
fi

STARTED=""
state() {
  python3 - "$H/state.db" <<'PY'
import sqlite3, json, sys
try:
    c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    rows = c.execute("select value from state_meta where key like 'goal:%'").fetchall()
    best = None
    for (v,) in rows:
        try:
            d = json.loads(v)
        except Exception:
            continue
        best = d
    if best:
        print(best.get("status", "?"), best.get("turns_used", "?"), best.get("max_turns", "?"), len(best.get("gates") or []))
    else:
        print("none 0 0 0")
except Exception as e:
    print("err", 0, 0, 0)
PY
}

for i in $(seq 1 30); do sleep 4; read -r ST TU MT NG <<< "$(state)"; [ "$ST" != none ] && [ "$ST" != err ] && { STARTED=1; break; }; done
GFAIL=""
if [ -n "$STARTED" ] && [ -n "${GATE_CMD:-}" ]; then
  # ordering check from the pane (turns_used is useless here: a fast model finishes turn 1 before the first poll):
  # the gate confirmation must sit between "Goal set" and the first goal-kickoff prompt line.
  GI=""; for i in $(seq 1 15); do tmux capture-pane -t "=$S:" -p -S -400 > "$B/pane-gatecheck.txt"; GI=$(grep -n "Gate added" "$B/pane-gatecheck.txt" | head -1 | cut -d: -f1); [ -n "$GI" ] && break; sleep 2; done
  SI=$(grep -n "Goal set" "$B/pane-gatecheck.txt" | head -1 | cut -d: -f1)
  KI=$(awk -v s="${SI:-0}" 'NR>s && /^❯ /{print NR; exit}' "$B/pane-gatecheck.txt")
  if [ -n "$GI" ] && { [ -z "$KI" ] || [ "$GI" -lt "$KI" ]; }; then say "quality gate registered before the first goal turn (pane lines: goal $SI, gate $GI, kickoff ${KI:-pending})";
  elif [ -n "$GI" ]; then GFAIL=1; say "WARN quality gate registered AFTER the first goal turn started (gate $GI, kickoff $KI)"; echo "{\"gate_late\": true}" > "$B/gate-registration.json";
  else GFAIL=1; say "WARN quality gate NOT registered"; echo "{\"gate_missing\": true}" > "$B/gate-registration.json"; fi
  # the goal text must not carry the gate line (the 71 failure mode: /goal gate add glued onto the goal text)
  GLUED=$(python3 - "$H/state.db" <<'PY'
import sqlite3, json, sys
try:
    c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    g = [json.loads(v).get("goal", "") for (v,) in c.execute("select value from state_meta where key like 'goal:%'")]
    print("yes" if any("/goal gate add" in (x or "") for x in g) else "no")
except Exception:
    print("err")
PY
)
  [ "$GLUED" = yes ] && { GFAIL=1; say "WARN goal text contains '/goal gate add' (input glued)"; echo "{\"goal_glued\": true}" > "$B/gate-registration.json"; }
  # VX_GATE_STRICT=1: a missing/late/glued gate ends the run here instead of burning the wall on a broken setup
  if [ "${VX_GATE_STRICT:-0}" = "1" ] && [ -n "$GFAIL" ]; then
    say "ABORT gate setup failed ($(cat "$B/gate-registration.json")), VX_GATE_STRICT=1"
    tmux capture-pane -t "=$S:" -p -S -2000 > "$B/pane-final.txt" 2>/dev/null
    tmux send-keys -t "=$S:" Escape; sleep 1; tmux send-keys -t "=$S:" -l "/exit"; tmux send-keys -t "=$S:" Enter; sleep 8
    tmux kill-session -t "=$S" 2>/dev/null
    touch "$B/pane-rec/STOP"; kill $RECPID 2>/dev/null
    echo "{\"home\":\"$H\",\"agent\":\"$AGENT\",\"slug\":\"$SLUG\",\"end\":\"gate_setup_failed\",\"wall_s\":$(( $(date +%s) - T0 ))}" > "$B/driver.json"
    exit 4
  fi
fi
say "goal registered: $ST $TU/$MT"
END=""
# stall nudge (HARDENING L37, 2026-10-29; opt-in VX_STALL_NUDGE=1, part of harness h3): when Hermes stops a looping response and goes
# back to its prompt with the goal still active, nothing continues (72a: 32 of 60 minutes idle). vx_stall.py = newest state.db message
# older than VX_STALL_SECS (default 720) AND the model server not processing. at most VX_STALL_MAX (3) nudges, >= 120 s apart, typed with
# type_line like any user line; count written to driver.json. fail-safe: a detector error prints `unknown` and nothing happens.
HERE=$(cd "$(dirname "$0")" && pwd); NUDGES=0; LAST_NUDGE=0
NUDGE_TEXT="Your previous response was cut off and discarded. Continue the goal from where you stopped: re-read your engine.py, fix whatever the quality gate or your own tests reported, and finish."
while :; do
  sleep 30
  NOW=$(date +%s); EL=$(( (NOW - T0) / 60 ))
  read -r ST TU MT NG <<< "$(state)"
  if [ "${VX_STALL_NUDGE:-0}" = "1" ] && [ "$ST" = active ]; then
    SV=$(python3 "$V/scripts/vx_stall.py" "$H/state.db" "${PORT}" "${VX_STALL_SECS:-720}" 2>/dev/null || echo "unknown detector-failed")
    case "$SV" in
      stall*) if [ "$NUDGES" -lt "${VX_STALL_MAX:-3}" ] && [ $(( NOW - LAST_NUDGE )) -gt 120 ]; then
                NUDGES=$(( NUDGES + 1 )); LAST_NUDGE=$NOW; say "STALL ($SV): nudge $NUDGES typed"; type_line "$NUDGE_TEXT"
              fi;;
    esac
  fi
  if tmux capture-pane -t "=$S:" -p -S -400 | grep -qE "Goal achieved"; then END="done"; fi
  if tmux capture-pane -t "=$S:" -p -S -400 | grep -qE "Goal paused|judged unachievable"; then END="paused"; fi
  case "$ST" in done|achieved) END="done";; paused|blocked|cleared) END="${END:-$ST}";; esac
  echo "[chess $SLUG] $(date +%H:%M:%S) t+${EL}m goal=$ST turns=$TU/$MT end=${END:-}" >> "$LOG"
  [ -n "$END" ] && break
  if [ "$EL" -ge "$WALL" ]; then END="wall_limit"; tmux send-keys -t "=$S:" Escape; sleep 2; tmux send-keys -t "=$S:" -l "/goal pause"; tmux send-keys -t "=$S:" Enter; sleep 20; break; fi
  if [ -n "${DEADLINE_EPOCH:-}" ] && [ "$NOW" -ge "$DEADLINE_EPOCH" ]; then END="hard_stop"; break; fi
  tmux has-session -t "=$S" 2>/dev/null || { END="hermes_exited"; break; }
done
WALL_S=$(( $(date +%s) - T0 ))
say "end=$END after ${WALL_S}s"
tmux capture-pane -t "=$S:" -p -S -2000 > "$B/pane-final.txt" 2>/dev/null
touch "$B/pane-rec/STOP"
tmux send-keys -t "=$S:" -l "/exit"; tmux send-keys -t "=$S:" Enter; sleep 8
tmux kill-session -t "=$S" 2>/dev/null
sleep 1; kill $RECPID 2>/dev/null
mkdir -p "$B/work"; cp -a "$W/." "$B/work/" 2>/dev/null
cp "$H/state.db" "$B/state.db" 2>/dev/null
# the home's own idle-exit gateway daemon (HARDENING L43): stop it by HERMES_HOME, never by name; the main gateway is untouched
HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME="$H" timeout 60 "$HBIN" gateway stop >> "$B/gateway-stop.log" 2>&1 || true
echo "{\"home\":\"$H\",\"agent\":\"$AGENT\",\"slug\":\"$SLUG\",\"model\":\"$MODEL\",\"ctx\":$CTX,\"port\":$PORT,\"end\":\"$END\",\"wall_s\":$WALL_S,\"stall_nudges\":$NUDGES}" > "$B/driver.json"
