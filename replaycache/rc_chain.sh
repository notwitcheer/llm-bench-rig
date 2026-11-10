#!/bin/bash
# rc_chain.sh [--nogpu] <ALL|B|C> <run_dir> <sel.json> <entries_dir>: the CPU legs of the replay-cache item. Runs inline in the item, or detached in tmux 96r-cpu
# (the --nogpu flag keeps the idle-queue runner from counting the session as a card owner). Every server here runs with CUDA_VISIBLE_DEVICES= and -ngl 0,
# nice 19, 6 threads, ports 8316 (1B) / 8317 (embedder), never :8090. Resumable: per-query result files are skipped.
#   B   : embed -> B (cache replay: best entry, the 1B pulls the numbers, solve() in a child)        -> <run>/cpu/B/*.json
#   C   : the 1B alone on every query (no cache)                                                       -> <run>/cpu/C/*.json, <run>/cpu/C.done
#   ALL : B, C, then summary -> render                                    -> summary.json, out/replay-cache.mp4, <run>/cpu/chain.done
# env: RC_DEADLINE (epoch; the legs stop starting new queries after it and the chain exits 3), RC_LIMITS (smoke/mock only: e.g. "--limit-templates 3 --limit-items 30").
set -uo pipefail
[ "${1:-}" = "--nogpu" ] && shift
MODE=${1:?mode ALL|B|C}; RUN=${2:?run dir}; SEL=${3:?sel}; ENT=${4:?entries}
RC=/home/witcheer/esoteric/replay-cache; CODE=$RC/code
SRV=/home/witcheer/llama.cpp-donald/build/bin/llama-server
M1B=$RC/models/Llama-3.2-1B-Instruct-Q8_0.gguf; MEMB=$RC/models/embeddinggemma-300M-Q8_0.gguf
P1B=${RC_P1B:-8316}; PEMB=${RC_PEMB:-8317}; THREADS=6
FF=/home/witcheer/voxelbench/ffmpeg; RPY=$RC/.venv/bin/python
LIM="${RC_LIMITS:-}"
DL=""; [ -n "${RC_DEADLINE:-}" ] && DL="--deadline $RC_DEADLINE"
export CUDA_VISIBLE_DEVICES= RC_DATA=$RC/data
mkdir -p "$RUN/cpu" "$RUN/out"
say() { echo "[rc-chain] $(date +%H:%M:%S) $*"; }
SP=""
stop_srv() { [ -n "$SP" ] || return 0; kill -TERM "$SP" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$SP" 2>/dev/null || break; sleep 1; done; kill -KILL "$SP" 2>/dev/null; SP=""; }
trap stop_srv EXIT
start_srv() { # <model> <port> <extra flags...>
  local m=$1 p=$2; shift 2
  nice -n 19 "$SRV" -m "$m" --host 127.0.0.1 --port "$p" -ngl 0 -t $THREADS "$@" > "$RUN/cpu/server-$p.log" 2>&1 &
  SP=$!
  local i H
  for i in $(seq 1 90); do H=$(curl -s -m 3 http://127.0.0.1:$p/health || true); [[ ${H,,} == *ok* ]] && { say "server :$p up (poll $i)"; return 0; }; kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  say "server :$p did not come up"; return 1
}
ARGS="--run $RUN --sel $SEL --entries $ENT --emb-base http://127.0.0.1:$PEMB --lm-base http://127.0.0.1:$P1B $LIM $DL"
say "mode $MODE (CUDA_VISIBLE_DEVICES empty, -ngl 0, $THREADS threads, nice 19)"
RC_B=0; RC_C=0
if [ "$MODE" = B ] || [ "$MODE" = ALL ]; then
  if [ ! -f "$RUN/cpu/B-embed.json" ]; then
    start_srv "$MEMB" $PEMB -c 2048 -np 1 --embedding --alias embed || exit 1
    nice -n 19 python3 "$CODE/rc_cpu.py" embed $ARGS || { say "embed failed"; exit 1; }
    stop_srv
  fi
  start_srv "$M1B" $P1B -c 4096 -np 1 --jinja --alias llama1b || exit 1
  nice -n 19 python3 "$CODE/rc_cpu.py" B $ARGS; RC_B=$?
  say "B rc=$RC_B"
  if [ "$MODE" = ALL ] && [ $RC_B = 0 ]; then
    nice -n 19 python3 "$CODE/rc_cpu.py" C $ARGS; RC_C=$?
    say "C rc=$RC_C"
  fi
  stop_srv
elif [ "$MODE" = C ]; then
  start_srv "$M1B" $P1B -c 4096 -np 1 --jinja --alias llama1b || exit 1
  nice -n 19 python3 "$CODE/rc_cpu.py" C $ARGS; RC_C=$?
  stop_srv
  say "C rc=$RC_C"
fi
[ $RC_C = 0 ] && [ "$MODE" != B ] && touch "$RUN/cpu/C.done"
[ $RC_B != 0 ] && exit $RC_B
[ $RC_C != 0 ] && exit $RC_C
[ "$MODE" = ALL ] || exit 0
[ -n "$LIM" ] && { say "limited run (smoke/mock): no summary, no render"; touch "$RUN/cpu/chain-limited.done"; exit 0; }
say "summary"
python3 "$CODE/rc_summary.py" "$RUN" "$SEL" > "$RUN/out/summary.stdout" 2>&1; src=$?
head -2 "$RUN/out/summary.stdout" | cut -c1-400
[ "$src" = 0 ] && [ -f "$RUN/summary.json" ] || { say "summary rc=$src, no summary.json"; exit $([ "$src" = 3 ] && echo 3 || echo 1); }
cp "$SEL" "$RUN/sel.json"
grep -m1 '^LINE1' "$RUN/out/summary.stdout" | cut -c7- > "$RUN/out/line1.txt"
if [ -z "${RC_NO_RENDER:-}" ]; then
  say "render"
  nice -n 19 "$RPY" "$CODE/rc_render.py" "$RUN" "$RUN/out/replay-cache.mp4" --ffmpeg "$FF" ${RC_MOCK_RENDER:+--mock} >> "$RUN/out/render.log" 2>&1 \
    && "$FF" -nostdin -y -loglevel error -ss 0.5 -i "$RUN/out/replay-cache.mp4" -frames:v 1 "$RUN/out/contact-1.png" </dev/null \
    && "$FF" -nostdin -y -loglevel error -ss 30 -i "$RUN/out/replay-cache.mp4" -frames:v 1 "$RUN/out/contact-2.png" </dev/null \
    && say "render ok" || say "render failed (see out/render.log); the verdict stands"
fi
touch "$RUN/cpu/chain.done"
say "chain done"
exit 0
