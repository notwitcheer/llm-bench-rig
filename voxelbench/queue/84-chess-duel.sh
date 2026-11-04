#!/bin/bash
# 84-chess-duel.sh (staged as 81, renamed 2026-11-03 so it sorts after 82): idle-queue item (enqueued 2026-11-03 on the operator yes; 2026-11-03, mercury subagent for the GPU thread; operator yes 2026-11-03 ~08:50 Paris on gpu-scout-2026-11-03 pick 2).
# WHAT: two local models each get ONE Hermes Agent /goal hour (60 min wall) to write a UCI chess engine in a single Python file from the same prompt (prompts/chess-v1.txt, standard library only,
#   python-chess and every other engine library forbidden and scanned for). Qwen3.8-27B Q6_K (the voxel series reference regime) goes first, then Ornith-1.5-35B Q4_K_M; one model on the card at a
#   time, Donald drained once and restored once. Then, CPU only and detached (tmux chess81cpu, any hour): a PRE-REGISTERED gate outside the agent's tree on each engine, a 20-game match engine vs
#   engine, an Elo anchor against a low-strength Stockfish, and the clip (board replay with clock + node counts, scoreboard still, 8x cutaway of the two Hermes panes).
# MOJO: two chess engines each coded by a local model in one Hermes /goal hour, then 20 games with the board replay, a clock and node counts | absurd + intense (the code fights, not the models; one engine may not know how to castle, may play an illegal move, may fail perft) | line 1 = the pre-registered variant below, e.g. "Qwen3.8-27B and Ornith-1.5-35B each wrote a UCI chess engine in one Hermes /goal hour on one RTX 5090. Both pass perft 1-5 and play legal chess. [X] won the 20-game match [a-b]."
# EV: line 1 as above; gpu-ev-gate-2026-10-25 section 2 rule 4 (reproducible, fully local Hermes piece) with two brand-pull combatants (Qwen, Ornith: our Ornith posts did 24,249 and 16,599 views, gate section 1); comparables: first voxel post 5,549, Hermes drives the duck 3,791, Hermes-on-local practical posts 4-14k, voxel sequels ~2k (so this is a NEW format with a scoreable result, not voxel again). bracket 3-10k. a gate FAIL is the headline, never hidden (variants V2/V3); X only with a watchable clip, otherwise TG + repo. hook check 2026-11-03 (not exhaustive): cloud agents have written engines (TeXCCChessEngine, a dev.to "5 AI agents" post), no local-model head-to-head with a perft gate found; never claim "first".
# serves: content (a Hermes-driven showcase with a scoreable result + a new format) + experiment (can a local model write a rules-correct engine in one hour: perft is a hard yes/no).
#
# PRE-REGISTERED (written 2026-11-03 before any agent run of this prompt; the final gate lives in ~/.vxfinal and is not named on any path the agent can read)
#  GATE, per engine (all must hold, in this order): the file compiles and imports only the standard library (no chess library, subprocess, network); `uci`/`isready` answered; `go movetime 500`
#   returns a legal move (referee's own rules) from four fixed positions; `go perft N` from the start position equals the published node count at depths 1 to 5 (reference numbers frozen from
#   chessprogramming.org/Perft_Results fetched 2026-11-03, timeouts 60/60/60/300/1,200 s); 10 games against a fixed weak baseline (a seeded random legal mover, 5 openings x 2 colours, 500 ms
#   per move) with NO illegal move, crash or late move. Reported, NOT gating: perft on four more positions (castling, en passant, promotion), W/D/L vs the baseline, perft seconds.
#   (the start position never castles or promotes in 5 plies: an engine can pass the gate and still not castle; the coverage extras say so.)
#  LINE-1 VARIANTS (build_summary.py fills the brackets from the data, nothing is worded by hand):
#   V1 both pass:  "<A> and <B> each wrote a UCI chess engine in one Hermes /goal hour on one RTX 5090. Both pass perft 1-5 and play legal chess. <match sentence>"
#   V2 one fails:  "<P> passes perft 1-5 and plays legal chess; <F> fails the gate: <first failed check and its number>. <match sentence if played>"
#   V3 both fail:  "Neither engine passes the gate: <A>: <check>. <B>: <check>. <match sentence if played>"
#   + a coverage sentence for a passing engine that fails a coverage extra ("passes the gate but fails the <position> coverage extra (<what it covers>)").
#  MATCH (played whenever both engines answer `uci`, gate verdict or not): 20 games, alternating colours, 10 fixed 4-ply openings x 2 colours, `go movetime 1000`, 1,500 ms grace, 240-ply
#   cap = draw, 4 games at a time on a 12-core CPU. illegal move / late move / crash = loss. points win 1, draw 0.5. RESULT RULE: points difference >= 3 "<W> won the match x-y"; 1 to 2.5 "<W>
#   edged it x-y (inside the noise of 20 games)"; 0 "level". No "model X is better at chess" claim is ever made from n = 1 engine per model. If an engine never answers `uci`: "no match".
#  ELO ANCHOR (skipped when no binary): each engine plays 10 games against Stockfish 19 (sf_19, GitHub release x86-64-universal, 81.4 MB, sha256 9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f)
#   limited with UCI_LimitStrength + UCI_Elo 1320 (its minimum), Threads 1, 1 s per move. Reported as "scored s/10 against Stockfish limited to 1320", never as a rating (the UCI_Elo scale is calibrated at a
#   longer time control; 10 games is +-200 Elo).
#  CONFOUNDS (label every number and the clip with them): Qwen3.8-27B Q6_K, temp 1.0 / top-p 0.95 / top-k 20 + min-p 0.05, MTP n-max 2 vs Ornith-1.5-35B-A3B Q4_K_M, temp 0.6 / top-p 0.95 / top-k 20 + min-p 0.05, no draft
#   (model-params.md rows, same files as 72fa / 72d); SAME: prompt, wall 60 min, 20 goal turns, ctx 131072 (fit ladder), q8_0 KV, flash-attn, --parallel 1, thinking cap 16,384 + budget message, llama.cpp 839a572,
#   harness h4 (isolated HERMES_HOME, quality gate registered before turn 1, stall nudge), toolsets terminal + file only. n = 1 run per model. speed (tok/s) is never a cross-model claim.
#  CARD: ~140 min (2 x (60 wall + ~5 load and teardown) + ~5 drain/restore); needs >= 165 min left in the window (one agent left over after a resume: >= 85) or exits 3 (stays queued). CPU phase after: ~60-90 min, detached.
#
# ISOLATION: each agent runs with its OWN HERMES_HOME (/tmp/chess81/home-<agent>-<epoch>, launched with the checkout's minted launcher, not a profile) and never touches the main hermes-gateway.service (the harness
#   stays isolated from the always-on gateway). the model server is on port 8581, never :8090 (Donald is drained with sudo -n systemctl stop and restored with start on EXIT). own-pid kills only, no pkill, no -f patterns.
# artifact: runs/<slug>/driver.json (exact). bench homes under /tmp/chess81 are throwaway; nothing under ~/.hermes/profiles is created.
set -uo pipefail
NL_TAG=chess81
MANIFEST=/home/witcheer/voxel-chess81-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
V=/home/witcheer/voxelbench
CD=$V/chess-duel
FX=/home/witcheer/.vxfinal/chess
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
RACEPY=/home/witcheer/scratch/race/venv/bin/python
SF=/home/witcheer/.vxfinal/anchor/stockfish/stockfish-linux-x86-64-universal
RUNLOG=/home/witcheer/voxel-pilot-run.log
ARENA=/tmp/chess81
PORT=8581
PROMPT=$CD/prompts/chess-v1.txt
QG=$CD/scripts/quick_gate_chess.py
DUEL=$V/runs/chess81-duel
CTX=131072; WALL=60; TURNS=20; KV=q8_0
RB=16384
RB_MSG="Thinking budget reached. Stop planning and write the files now with your tools; refine in later turns."
# agent A: Qwen3.8-27B Q6_K, the series reference (same server regime as 72fa)
A_SLUG=chess81-qwen38-27b-q6k; A_AGENT=qwen; A_NAME="Qwen3.8-27B"
A_FILE=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf; A_BYTES=22884408288; A_ALIAS=Qwen3.8-27B-Q6_K.gguf
A_SAMP="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"; A_EXTRA=(--spec-type draft-mtp --spec-draft-n-max 2)
A_PARAMS="GGUF general.sampling keys of the maker file (temp 1.0, top_p 0.95, top_k 20), voxelbench/model-params.md row Qwen3.8-27B Q6_K; no min_p key, so llama.cpp default 0.05; RB 16384 (series value)"
A_REGIME="Q6_K, ctx 128k, q8_0 KV, MTP n2, temp 1.0"
# agent B: Ornith-1.5-35B-A3B Q4_K_M (same server regime as 72d)
B_SLUG=chess81-ornith15-35b-q4km; B_AGENT=ornith; B_NAME="Ornith-1.5-35B"
B_FILE=/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf; B_BYTES=21713462848; B_ALIAS=Ornith-1.5-35B-Q4_K_M.gguf
B_SAMP="--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05"; B_EXTRA=()
B_PARAMS="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF fetched 2026-10-28 (general tasks: temperature=0.6, top_p=0.95, top_k=20; no output budget named, RB floor 16384), voxelbench/model-params.md row Ornith 1.5 35B-A3B"
B_REGIME="Q4_K_M, ctx 128k, q8_0 KV, no draft, temp 0.6"
# the final gate's file names are assembled so this item (it sits on a path the agent can read) never carries them
GATEF="$FX/chess_final_""gate.py"; BASEF="$FX/baseline_""random.py"; REFF="$FX/perft_""ref.json"; LIBF="$FX/chess""lib.py"

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$BIN" "$GPY" "$RACEPY" "$A_FILE" "$B_FILE" "$V/scripts/vx_stall.py" "$V/scripts/metrics.py" "$PROMPT" "$QG" "$CD/scripts/qgate_data.json" "$CD/scripts/chess_agent.sh" \
  "$CD/scripts/chess_mkhome.py" "$CD/scripts/pane_rec.py" "$GATEF" "$BASEF" "$REFF" "$LIBF" "$FX/chess""match.py" "$FX/match_run.py" "$FX/build_summary.py" "$FX/cpu_phase.sh" "$FX/render/chess_render.py" \
  "$FX/tests/test_chess""lib.py" "$V/ffmpeg" /home/witcheer/.hermes/hermes-agent/.hermes/bin/hermes
[ -x "$SF" ] && nl_say "  ok: Elo anchor binary $SF" || nl_say "  note: no Stockfish binary at $SF: the anchor step is skipped (not a failure)"
for pair in "A:$A_FILE:$A_BYTES" "B:$B_FILE:$B_BYTES"; do IFS=: read -r t f n <<< "$pair"; SZ=$(stat -c %s "$f" 2>/dev/null || echo 0); [ "$SZ" = "$n" ] && nl_say "  ok: model $t $SZ bytes" || nl_fail "model $t size $SZ != $n"; done
$BIN --version 2>&1 | grep -q "839a572" && nl_say "  ok: llama.cpp 839a572" || { nl_say "  llama.cpp build changed or missing"; exit 2; }
HELP=$($BIN --help 2>&1 || true); case "$HELP" in *--reasoning-budget-message*) nl_say "  ok: --reasoning-budget(-message) in build";; *) nl_fail "--reasoning-budget-message missing from build";; esac
for f in $A_SAMP $B_SAMP "${A_EXTRA[@]}"; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this build";; esac;; esac; done; nl_say "  ok: sampling/extra flags exist in build: $A_SAMP | $B_SAMP | ${A_EXTRA[*]}"
# L33 enforcement: the final gate's file names and numbers must not sit on any path the agent can read (the visible harness tree and every queued item). assembled from pieces so this scan does not match itself.
LEAKS=$(grep -rIl --exclude-dir=runs --exclude-dir=deb --exclude-dir=extract --exclude-dir=lib --exclude-dir=previews --exclude-dir=__pycache__ --exclude="runs-*" \
  -e "chess_final_""gate" -e "perft_""ref" -e "baseline_""random" -e "kiwi""pete" -e "4865""609" -e "4,865,""609" -e "chess""lib" -e "chess""match" "$V" /home/witcheer/idle-queue/*.sh 2>/dev/null || true)
[ -z "$LEAKS" ] || nl_fail "final-gate names or numbers visible to the agent in: $(echo $LEAKS | tr '\n' ' ')"
nl_say "  ok: final-gate names and numbers not visible on the agent paths ($V, ~/idle-queue/*.sh); final gate dir $FX"
MG=$(grep -rIl -e "def legal_""moves" -e "def pseudo_""moves" -e "def gen_""moves" "$CD" 2>/dev/null || true)
[ -z "$MG" ] && nl_say "  ok: no move generator in the visible chess-duel tree ($CD)" || nl_fail "a move generator sits in the visible chess-duel tree: $MG"
"$GPY" -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$QG" "$CD/scripts/chess_mkhome.py" "$CD/scripts/pane_rec.py" "$V/scripts/vx_stall.py" "$GATEF" "$FX/chess""match.py" "$FX/match_run.py" "$FX/build_summary.py" "$FX/render/chess_render.py" || nl_fail "script syntax"
bash -n "$CD/scripts/chess_agent.sh" && bash -n "$FX/cpu_phase.sh" || nl_fail "shell syntax"
"$RACEPY" -c "import PIL, pyte, fontTools" 2>/dev/null && nl_say "  ok: render python has PIL, pyte, fontTools" || nl_fail "render python lacks PIL/pyte/fontTools"
# the referee must still reproduce the frozen node counts right before the run (CPU, ~3 s)
python3 "$FX/tests/test_chess""lib.py" > /tmp/chess81-reftest.txt 2>&1 && nl_say "  ok: referee rules reproduce the frozen reference counts ($(tail -1 /tmp/chess81-reftest.txt))" || nl_fail "referee self-test failed: $(tail -2 /tmp/chess81-reftest.txt | tr '\n' ' ')"
ss -ltnH "sport = :$PORT" | grep -q . && nl_fail "port $PORT is already in use by someone else"
for s in chess81-qwen chess81-ornith chess81cpu; do tmux has-session -t "=$s" 2>/dev/null && nl_fail "tmux session $s already exists"; done
GG_A=$($GPY - "$A_FILE" <<'PYEOF' 2>/dev/null
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
nl_say "  params A: src='$A_PARAMS' gguf_sampling='${GG_A:-none}' flags='$A_SAMP ${A_EXTRA[*]}'"
nl_say "  params B: src='$B_PARAMS' flags='$B_SAMP'"
nl_check_disk 20
RUN_A=1; RUN_B=1
if nl_skip_exact "$V/runs/$A_SLUG/driver.json"; then nl_say "  SKIP: $A_SLUG artifact exists: $V/runs/$A_SLUG/driver.json"; RUN_A=0; else nl_say "  RUN: $A_SLUG artifact $V/runs/$A_SLUG/driver.json (engine $V/runs/$A_SLUG/work/engine.py)"; fi
if nl_skip_exact "$V/runs/$B_SLUG/driver.json"; then nl_say "  SKIP: $B_SLUG artifact exists: $V/runs/$B_SLUG/driver.json"; RUN_B=0; else nl_say "  RUN: $B_SLUG artifact $V/runs/$B_SLUG/driver.json (engine $V/runs/$B_SLUG/work/engine.py)"; fi
nl_say "  RUN: cpu phase (detached tmux chess81cpu) artifact $DUEL/CPU_PHASE_DONE (gates $DUEL/gate-a/gate.json $DUEL/gate-b/gate.json, match $DUEL/match/match.json, line 1 $DUEL/line1.txt, clips $DUEL/replay.mp4 $DUEL/cutaway.mp4 $DUEL/scoreboard.png)"
N_RUN=$((RUN_A + RUN_B))
[ "$N_RUN" -eq "${EXPECT_RUN:-$N_RUN}" ] || { nl_say "PLAN: planned RUN $N_RUN != EXPECT_RUN ${EXPECT_RUN:-}; nothing started"; exit 2; }
DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
NEED=165; [ "$N_RUN" -le 1 ] && NEED=85
nl_say "  planned: $N_RUN agent hour(s) (wall $WALL min each + load/teardown ~5 min), card ~$((N_RUN*65+5)) min; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away, need >= $NEED"
if [ "$N_RUN" -eq 0 ]; then nl_say "both agent legs done: only the cpu phase can remain (see $DUEL/CPU_PHASE_DONE)"; exit 0; fi
[ "$LEFT" -ge "$NEED" ] || { nl_say "  under $NEED min left in the window: not starting (two agent hours + load need >= 165; one needs >= 85; exit 3, resumable)"; exit 3; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "$A_SLUG" "$B_SLUG" "cpu-phase"; fi
nl_gate

# ---------------------------------------------------------------- RUN
SRV=""; DONALD_DRAINED=0
killsrv() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; SRV=""; }   # only the server THIS item started
restore_donald() {
  [ "$DONALD_DRAINED" = 1 ] || return 0
  nl_say "restoring Donald (sudo -n systemctl start llama-server.service)"
  sudo -n systemctl start llama-server.service
  for i in $(seq 1 150); do curl -s -m 3 127.0.0.1:8090/health | grep -qi ok && { nl_say "  Donald healthy again after ~$((i*2)) s"; DONALD_DRAINED=0; return 0; }; sleep 2; done
  nl_say "  WARN Donald not healthy 5 min after start (check: systemctl status llama-server.service)"; DONALD_DRAINED=0
}
cleanup() { killsrv; for a in $A_AGENT $B_AGENT; do tmux kill-session -t "=chess81-$a" 2>/dev/null; done; sleep 3; restore_donald; }
trap cleanup EXIT
mkdir -p "$DUEL" "$ARENA"
nl_say "draining Donald"; DONALD_DRAINED=1; sudo -n systemctl stop llama-server.service; sleep 5
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "${v:-99999}" -lt 1024 ] || { nl_say "Donald drain never cleared (${v} MiB): aborting before any load"; nl_mark "$A_SLUG" failed "drain never cleared"; exit 1; }

agent_leg() { # <slug> <agent> <name> <model_file> <alias> <params_src> <samp> then EXTRA_ARR global, <deadline_epoch>
  local slug=$1 agent=$2 name=$3 mfile=$4 alias=$5 psrc=$6 samp=$7 dl=$8
  local t0 UP=0 c CTX_USED
  mkdir -p "$V/runs/$slug"
  [ "$(ls -A $V/runs/$slug/work 2>/dev/null)" ] && { nl_say "  stale work dir from an earlier attempt: moving aside"; mv "$V/runs/$slug" "$V/runs/$slug.stale-$(date +%s)"; mkdir -p "$V/runs/$slug"; }
  t0=$(date +%s); nl_mark "$slug" running
  for c in $CTX 98304 65536; do
    [ "$c" -le "$CTX" ] || continue
    # shellcheck disable=SC2086
    $BIN --model "$mfile" --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size $c --parallel 1 --flash-attn on \
      --cache-type-k $KV --cache-type-v $KV --alias "$alias" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $samp "${EXTRA_ARR[@]}" \
      > "$V/runs-$slug-server.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 150); do curl -s 127.0.0.1:$PORT/health | grep -qi ok && { UP=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 2; done
    [ "$UP" = 1 ] && { CTX_USED=$c; break; }
    nl_say "  server did not come up at ctx $c: $(grep -m1 -iE 'out of memory|failed|error|unknown' "$V/runs-$slug-server.log" | head -c 200)"
    killsrv; sleep 4
  done
  [ "$UP" = 1 ] || { nl_mark "$slug" failed "server did not come up at any ctx (see runs-$slug-server.log)"; echo '{"end": "server_failed", "wall_s": 0}' > "$V/runs/$slug/driver.json"; return 1; }
  nl_say "$name server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX_USED (requested $CTX)"
  curl -s 127.0.0.1:$PORT/props > "$V/runs/$slug/props.json" 2>/dev/null
  nl_say "  server sampling defaults: $(python3 -c "import json;g=json.load(open('$V/runs/$slug/props.json')).get('default_generation_settings',{});p=g.get('params',g);print({k:p.get(k) for k in ('temperature','top_k','top_p','min_p','n_predict')})" 2>/dev/null)"
  echo "{\"model_file\": \"$mfile\", \"ctx\": $CTX_USED, \"kv\": \"$KV\", \"extra\": \"${EXTRA_ARR[*]}\", \"sampling\": \"$samp\", \"params_src\": \"$psrc\", \"reasoning_budget\": $RB, \"prompt\": \"chess-v1.txt\", \"port\": $PORT, \"build\": \"839a572a4\", \"vram_used\": \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)\"}" > "$V/runs/$slug/run-config.json"
  local W_DIR=$ARENA/$agent
  { cat "$PROMPT"; printf '\nThe working directory is %s. The final file must be exactly %s/engine.py (copy this path, never retype it).\n' "$W_DIR" "$W_DIR"; } > "$V/runs/$slug/prompt.txt"
  export DEADLINE_EPOCH=$dl
  PORT=$PORT ARENA=$ARENA V=$V VX_GATE_STRICT=1 VX_STALL_NUDGE=1 GATE_CMD="python3 $QG $W_DIR/engine.py $V/runs/$slug/qgate" \
    bash "$CD/scripts/chess_agent.sh" "$slug" "$agent" "$alias" "$CTX_USED" "$WALL" "$TURNS" "$V/runs/$slug/prompt.txt" >> "$RUNLOG" 2>&1
  nl_say "driver: $(cat "$V/runs/$slug/driver.json" 2>/dev/null)"
  curl -s 127.0.0.1:$PORT/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > "$V/runs/$slug/server-metrics.txt" 2>/dev/null
  killsrv; sleep 3
  python3 "$V/scripts/metrics.py" "$(python3 -c "import json;print(json.load(open('$V/runs/$slug/driver.json'))['home'])" 2>/dev/null)" "$V/runs/$slug/metrics.json" >> "$RUNLOG" 2>&1
  # the card must be empty before the next model loads
  for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1500 ] && break; sleep 2; done
  if [ -f "$V/runs/$slug/work/engine.py" ]; then nl_say "  $name wrote engine.py: $(wc -c < "$V/runs/$slug/work/engine.py") bytes, $(wc -l < "$V/runs/$slug/work/engine.py") lines"; else nl_say "  $name left NO engine.py (the gate reports it)"; fi
  nl_judge_artifact "$slug" 0 "$V/runs/$slug/driver.json" "$t0" "$(( ($(date +%s)-t0)/60 )) min" || true
  return 0
}

DEADLINE=$(nl_deadline_epoch)
if [ "$RUN_A" = 1 ]; then
  EXTRA_ARR=("${A_EXTRA[@]}")
  DL_A=$(( DEADLINE - 300 )); [ "$RUN_B" = 1 ] && DL_A=$(( DEADLINE - 70*60 ))   # leave room for the second hour
  agent_leg "$A_SLUG" "$A_AGENT" "$A_NAME" "$A_FILE" "$A_ALIAS" "$A_PARAMS" "$A_SAMP" "$DL_A"
fi
if [ "$RUN_B" = 1 ]; then
  if [ "$(( DEADLINE - $(date +%s) ))" -lt $(( 75*60 )) ]; then nl_say "less than 75 min left after agent A: agent B stays queued (exit 3 after the cpu phase for A)"; B_SKIPPED=1; else
    EXTRA_ARR=("${B_EXTRA[@]}")
    agent_leg "$B_SLUG" "$B_AGENT" "$B_NAME" "$B_FILE" "$B_ALIAS" "$B_PARAMS" "$B_SAMP" "$(( DEADLINE - 300 ))"
  fi
fi
killsrv; restore_donald
# ---------------------------------------------------------------- CPU phase (detached, any hour): only when BOTH agent legs exist
if [ -f "$V/runs/$A_SLUG/driver.json" ] && [ -f "$V/runs/$B_SLUG/driver.json" ]; then
  nl_say "launching the CPU phase in tmux chess81cpu (gates, 20-game match, Elo anchor, clips); log $DUEL/cpu-phase.log"
  tmux new-session -d -s chess81cpu "nogpu=1 AGENT_DIR_A=$V/runs/$A_SLUG AGENT_DIR_B=$V/runs/$B_SLUG REGIME_A='$A_REGIME' REGIME_B='$B_REGIME' SF=$SF PYR=$RACEPY FF=$V/ffmpeg bash $FX/cpu_phase.sh $DUEL '$A_NAME' '$B_NAME' $V/runs/$A_SLUG/work/engine.py $V/runs/$B_SLUG/work/engine.py >> $DUEL/cpu-phase.out 2>&1"
  sleep 3; tmux has-session -t "=chess81cpu" 2>/dev/null && nl_say "  chess81cpu running" || nl_say "  chess81cpu did not start: run by hand: nogpu=1 AGENT_DIR_A=$V/runs/$A_SLUG AGENT_DIR_B=$V/runs/$B_SLUG bash $FX/cpu_phase.sh $DUEL '$A_NAME' '$B_NAME' $V/runs/$A_SLUG/work/engine.py $V/runs/$B_SLUG/work/engine.py"
  nl_mark "cpu-phase" launched "tmux chess81cpu, log $DUEL/cpu-phase.out"
fi
[ "${B_SKIPPED:-0}" = 1 ] && exit 3
exit 0
