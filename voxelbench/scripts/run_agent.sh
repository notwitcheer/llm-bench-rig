#!/bin/bash
# voxel bench agent driver (capsule). assumes an OpenAI-compatible server is already serving <model_id> on :8090.
# usage: run_agent.sh <slug> <model_id> <ctx> <wall_min> <goal_turns> <prompt_file>
# runs Hermes (classic CLI) in tmux with an ISOLATED HERMES_HOME, sets the task with /goal, polls the goal state in
# that home's state.db until done/paused/blocked or the wall limit, exits, then the caller runs the gate.
set -uo pipefail
SLUG=$1; MODEL=$2; CTX=$3; WALL=$4; TURNS=$5; PROMPT_FILE=$6
B=/home/witcheer/voxelbench/runs/$SLUG
P=vx$(echo "$SLUG" | tr -cd "a-z0-9")
H=/home/witcheer/.hermes/profiles/$P; W=$B/work; LOG=$B/driver.log
HERMES=/home/witcheer/.local/bin/hermes
mkdir -p "$B" "$W"
[ -d "$H" ] || /home/witcheer/.local/bin/hermes profile create "$P" --no-alias --description "voxel bench run $SLUG (isolated, mercury)" >> "$B/profile.log" 2>&1
[ -d "$H" ] || { echo "profile create failed"; exit 2; }
say() { echo "[voxel $SLUG] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

cat > "$H/config.yaml" <<EOF
model:
  default: $MODEL
  provider: custom
  base_url: http://127.0.0.1:${PORT:-8090}/v1
  api_key: local-key
  context_length: $CTX
toolsets:
  - hermes-cli
agent:
  max_turns: 90
  clarify_timeout: 20
approvals:
  mode: "off"
memory:
  memory_enabled: false
  user_profile_enabled: false
goals:
  max_turns: $TURNS
display:
  interface: cli
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
EOF
cat > "$H/SOUL.md" <<'EOF'
You are a capable autonomous coding agent. Work without asking the user questions: nobody is watching this session.
EOF

GOAL=$(tr '\n' ' ' < "$PROMPT_FILE" | sed 's/  */ /g')
S="voxel-$SLUG"
tmux kill-session -t "=$S" 2>/dev/null
# socket-safe temp dir (2026-10-22, HARDENING L9): Hermes points TMPDIR at <profile>/cache/scratch for terminal commands;
# with our long bench profile names that path is ~76 chars and headless Chromium's AF_UNIX socket under it exceeds the
# 108-byte limit, so a model could never launch a browser from its terminal (TargetClosedError; <= 60 chars works).
# Hermes respects a TMPDIR set by the user, so VX_TMPFIX=1 starts Hermes with a short one. default 0 keeps the queued
# one-variable reruns (66z, 66zy) on the same harness as every earlier run; day0-voxel.sh and new items set 1.
TMPENV=""
if [ "${VX_TMPFIX:-0}" = "1" ]; then T=/tmp/vxt-$(echo -n "$SLUG" | md5sum | cut -c1-8); mkdir -p "$T"; TMPENV="TMPDIR=$T TMP=$T TEMP=$T "; echo "[voxel $SLUG] $(date +%T) socket-safe TMPDIR $T" >> "$LOG"; fi
tmux new-session -d -s "$S" -x 200 -y 50 "cd $W && ${TMPENV}$HERMES -p $P"
# wait for the prompt to be ready
for i in $(seq 1 60); do sleep 2; tmux capture-pane -t "=$S:" -p | grep -qE "❯|›|> *$" && break; done
sleep 3
tmux capture-pane -t "=$S:" -p > "$B/pane-start.txt"
say "hermes up, sending /goal (turn budget $TURNS, wall $WALL min, ctx $CTX, model $MODEL)"
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
  # lines (they queue), then SIGCONT. the server only pauses for ~4 s; a trap guarantees the CONT.
  SPID=$(ss -ltnpH "sport = :${PORT:-8090}" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$SPID" ] && { trap 'kill -CONT '"$SPID"' 2>/dev/null' EXIT; kill -STOP "$SPID" 2>/dev/null && say "model server pid $SPID held for the gate setup"; } || say "WARN no server pid on port ${PORT:-8090}: gate setup relies on timing"
  tmux send-keys -t "=$S:" -l "Reply with the single word ready and nothing else."; sleep 0.5; tmux send-keys -t "=$S:" Enter
  sleep 1.5
  tmux send-keys -t "=$S:" -l "/goal $GOAL"; sleep 0.5; tmux send-keys -t "=$S:" Enter
  sleep 0.8
  tmux send-keys -t "=$S:" -l "/goal gate add $GATE_CMD"; sleep 0.5; tmux send-keys -t "=$S:" Enter
  sleep 1.2
  [ -n "$SPID" ] && { kill -CONT "$SPID" 2>/dev/null; trap - EXIT; say "model server released"; }
else
  tmux send-keys -t "=$S:" -l "/goal $GOAL"
  sleep 1
  tmux send-keys -t "=$S:" Enter
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
if [ -n "$STARTED" ] && [ -n "${GATE_CMD:-}" ]; then
  # ordering check from the pane (turns_used is useless here: a fast model finishes turn 1 before the first poll):
  # the gate confirmation must sit between "Goal set" and the first goal-kickoff prompt line.
  GI=""; for i in $(seq 1 15); do tmux capture-pane -t "=$S:" -p -S -400 > "$B/pane-gatecheck.txt"; GI=$(grep -n "Gate added" "$B/pane-gatecheck.txt" | head -1 | cut -d: -f1); [ -n "$GI" ] && break; sleep 2; done
  SI=$(grep -n "Goal set" "$B/pane-gatecheck.txt" | head -1 | cut -d: -f1)
  KI=$(awk -v s="${SI:-0}" 'NR>s && /^❯ /{print NR; exit}' "$B/pane-gatecheck.txt")
  if [ -n "$GI" ] && { [ -z "$KI" ] || [ "$GI" -lt "$KI" ]; }; then say "quality gate registered before the first goal turn (pane lines: goal $SI, gate $GI, kickoff ${KI:-pending})";
  elif [ -n "$GI" ]; then say "WARN quality gate registered AFTER the first goal turn started (gate $GI, kickoff $KI)"; echo "{\"gate_late\": true}" > "$B/gate-registration.json";
  else say "WARN quality gate NOT registered"; echo "{\"gate_missing\": true}" > "$B/gate-registration.json"; fi
fi
say "goal registered: $ST $TU/$MT"
END=""
while :; do
  sleep 30
  NOW=$(date +%s); EL=$(( (NOW - T0) / 60 ))
  read -r ST TU MT NG <<< "$(state)"
  if tmux capture-pane -t "=$S:" -p -S -400 | grep -qE "Goal achieved"; then END="done"; fi
  if tmux capture-pane -t "=$S:" -p -S -400 | grep -qE "Goal paused|judged unachievable"; then END="paused"; fi
  case "$ST" in done|achieved) END="done";; paused|blocked|cleared) END="${END:-$ST}";; esac
  echo "[voxel $SLUG] $(date +%H:%M:%S) t+${EL}m goal=$ST turns=$TU/$MT end=${END:-}" >> "$LOG"
  [ -n "$END" ] && break
  if [ "$EL" -ge "$WALL" ]; then END="wall_limit"; tmux send-keys -t "=$S:" Escape; sleep 2; tmux send-keys -t "=$S:" -l "/goal pause"; tmux send-keys -t "=$S:" Enter; sleep 20; break; fi
  if [ -n "${DEADLINE_EPOCH:-}" ] && [ "$NOW" -ge "$DEADLINE_EPOCH" ]; then END="hard_stop"; break; fi
  tmux has-session -t "=$S" 2>/dev/null || { END="hermes_exited"; break; }
done
WALL_S=$(( $(date +%s) - T0 ))
say "end=$END after ${WALL_S}s"
tmux capture-pane -t "=$S:" -p -S -2000 > "$B/pane-final.txt" 2>/dev/null
tmux send-keys -t "=$S:" -l "/exit"; tmux send-keys -t "=$S:" Enter; sleep 8
tmux kill-session -t "=$S" 2>/dev/null
echo "{\"profile\":\"$P\",\"slug\":\"$SLUG\",\"model\":\"$MODEL\",\"ctx\":$CTX,\"end\":\"$END\",\"wall_s\":$WALL_S}" > "$B/driver.json"
