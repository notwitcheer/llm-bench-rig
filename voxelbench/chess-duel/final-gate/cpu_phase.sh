#!/bin/bash
# cpu_phase.sh: everything after the two agent hours, CPU only (any hour, unrestricted by the heures creuses rule, niced). Resumable: a step whose artefact exists is skipped (FORCE=1 redoes all).
# usage: cpu_phase.sh <root> <name-a> <name-b> <engine-a.py> <engine-b.py>
#   env: AGENT_DIR_A / AGENT_DIR_B (run dirs holding driver.json, metrics.json, work/engine.py, pane-rec/frames.jsonl; symlinked as <root>/agent-a|b)
#        REGIME_A / REGIME_B (text under the names on the scoreboard)   MOVETIME (1000)  GAMES (20)  CONC (4)  ANCHOR (1)  SF (stockfish binary)  PYR (python with PIL+pyte+fontTools)
#        GATE_* env passes through to chess_final_gate.py (tests only)   SKIP_RENDER=1  FORCE=1
# steps: 1 final gate on each engine  2 the 20-game match (when both answer `uci`)  3 Elo anchor vs Stockfish UCI_Elo 1320 (10 games each, when a binary exists)
#        4 summary + the pre-registered line 1  5 renders: scoreboard.png, replay.mp4, cutaway.mp4 (8x Hermes panes), contact sheets  6 leak scan of every published artefact
set -uo pipefail
ROOT=$1; NA=$2; NB=$3; EA=$4; EB=$5
FX=$(cd "$(dirname "$0")" && pwd)
MOVETIME=${MOVETIME:-1000}; GAMES=${GAMES:-20}; CONC=${CONC:-4}; ANCHOR=${ANCHOR:-1}
SF=${SF:-/home/witcheer/.vxfinal/anchor/stockfish/stockfish-linux-x86-64-universal}
PYR=${PYR:-/home/witcheer/scratch/race/venv/bin/python}
FF=${FF:-/home/witcheer/voxelbench/ffmpeg}
LOG=$ROOT/cpu-phase.log
mkdir -p "$ROOT"
say() { echo "[chess81-cpu] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
step() { # <artefact> -> 0 = run it
  [ "${FORCE:-0}" = 1 ] && return 0; [ -e "$1" ] && { say "skip (exists): $1"; return 1; }; return 0; }
for k in a b; do d=AGENT_DIR_${k^^}; [ -n "${!d:-}" ] && [ ! -e "$ROOT/agent-$k" ] && ln -s "${!d}" "$ROOT/agent-$k"; done
say "start: $NA vs $NB, movetime $MOVETIME ms, $GAMES games, conc $CONC, anchor=$ANCHOR"
# ---- 1 gates
for k in a b; do
  E=$EA; [ $k = b ] && E=$EB
  if step "$ROOT/gate-$k/gate.json"; then
    if [ -f "$E" ]; then nice -n 10 python3 "$FX/chess_final_gate.py" "$E" "$ROOT/gate-$k" >> "$ROOT/gate-$k.log" 2>&1; say "gate $k rc=$?: $(tail -1 "$ROOT/gate-$k.log")"
    else mkdir -p "$ROOT/gate-$k"; echo '{"pass": false, "no_engine": true, "checks": {"uci_handshake": false}}' > "$ROOT/gate-$k/gate.json"; say "gate $k: no engine.py at $E"; fi
  fi
done
# ---- 2 match (both engines must answer uci)
hs() { python3 -c "import json,sys;g=json.load(open(sys.argv[1]));sys.exit(0 if g.get('checks',{}).get('uci_handshake') else 1)" "$ROOT/gate-$1/gate.json"; }
if hs a && hs b; then
  if step "$ROOT/match/match.json"; then nice -n 10 python3 "$FX/match_run.py" duel "$ROOT/match" --a "$EA" --a-name "$NA" --b "$EB" --b-name "$NB" --games "$GAMES" --movetime "$MOVETIME" --conc "$CONC" >> "$ROOT/match.log" 2>&1; say "match rc=$?: $(tail -1 "$ROOT/match.log")"; fi
else say "match not played: an engine never answered uci"; fi
# ---- 3 anchor
if [ "$ANCHOR" = 1 ] && [ -x "$SF" ]; then
  for k in a b; do
    E=$EA; N=$NA; [ $k = b ] && { E=$EB; N=$NB; }
    if hs $k && step "$ROOT/anchor-$k/anchor.json"; then nice -n 10 python3 "$FX/match_run.py" anchor "$ROOT/anchor-$k" --a "$E" --a-name "$N" --sf "$SF" --games "${ANCHOR_GAMES:-10}" --movetime "$MOVETIME" --conc 3 >> "$ROOT/anchor-$k.log" 2>&1; say "anchor $k rc=$?: $(tail -1 "$ROOT/anchor-$k.log")"; fi
  done
else say "anchor skipped (ANCHOR=$ANCHOR, stockfish binary $([ -x "$SF" ] && echo present || echo missing))"; fi
# ---- 4 summary
python3 "$FX/build_summary.py" "$ROOT" --a-name "$NA" --b-name "$NB" --a-regime "${REGIME_A:-}" --b-regime "${REGIME_B:-}" --movetime "$MOVETIME" >> "$LOG" 2>&1; say "summary: $(cat "$ROOT/line1.txt" 2>/dev/null)"
# ---- 5 renders
if [ "${SKIP_RENDER:-0}" != 1 ]; then
  [ -f "$ROOT/scoreboard.json" ] && step "$ROOT/scoreboard.png" && "$PYR" "$FX/render/chess_render.py" scoreboard "$ROOT/scoreboard.json" "$ROOT/scoreboard.png" >> "$LOG" 2>&1
  if [ -f "$ROOT/match/match.json" ] && step "$ROOT/replay.mp4"; then
    FEAT=$("$PYR" - "$ROOT/match/match.json" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))["games"]
# featured game = the longest decisive game (a game that ends), else game 1
dec = [(len(x["moves"]), x["round"]) for x in g if x["result"] != "1/2-1/2" and not (x["reason"] or "").startswith("adjudicated")]
print(max(dec)[1] if dec else 1)
PYEOF
)
    FFMPEG=$FF nice -n 10 "$PYR" "$FX/render/chess_render.py" replay "$ROOT/match/match.json" "$ROOT/replay.mp4" --name-a "$NA" --name-b "$NB" --game "${FEAT:-1}" --scoreboard-json "$ROOT/scoreboard.json" >> "$LOG" 2>&1; say "replay.mp4 rc=$? (featured game ${FEAT:-1})"
  fi
  if [ -f "$ROOT/agent-a/pane-rec/frames.jsonl" ] && [ -f "$ROOT/agent-b/pane-rec/frames.jsonl" ] && step "$ROOT/cutaway.mp4"; then
    FFMPEG=$FF nice -n 10 "$PYR" "$FX/render/chess_render.py" cutaway "$ROOT/agent-a/pane-rec/frames.jsonl" "$ROOT/agent-b/pane-rec/frames.jsonl" "$ROOT/cutaway.mp4" --name-a "$NA" --name-b "$NB" --speed 8 >> "$LOG" 2>&1; say "cutaway.mp4 rc=$?"
  fi
  mkdir -p "$ROOT/sheets"
  for m in replay cutaway; do
    [ -f "$ROOT/$m.mp4" ] || continue
    dur=$("$FF" -i "$ROOT/$m.mp4" 2>&1 | grep -oE 'Duration: [0-9:.]+' | cut -d' ' -f2 | awk -F: '{print $1*3600+$2*60+$3}')
    "$FF" -nostdin -v error -y -i "$ROOT/$m.mp4" -vf "fps=8/${dur:-30},scale=640:-2,tile=4x2" -frames:v 1 "$ROOT/sheets/$m.png" < /dev/null && say "contact sheet $m: $ROOT/sheets/$m.png"
  done
fi
# ---- 6 leak scan of the artefacts that can be published (text files only; the mp4s are read from the contact sheets by eye)
LEAK=$(grep -rIlE "/home/|/Users/|witcheer|100\.[0-9]+\.[0-9]+\.[0-9]+|\.ts\.net" "$ROOT"/match/match.pgn "$ROOT"/match/match.json "$ROOT"/anchor-*/anchor.pgn "$ROOT"/summary.json "$ROOT"/scoreboard.json "$ROOT"/result.md "$ROOT"/line1.txt "$ROOT"/gate-*/gate.json 2>/dev/null | tr '\n' ' ')
[ -z "$LEAK" ] && say "leak scan of published text artefacts: clean" || say "LEAK SCAN HITS (fix before anything leaves capsule): $LEAK"
date -u +%FT%TZ > "$ROOT/CPU_PHASE_DONE"
say "done"
