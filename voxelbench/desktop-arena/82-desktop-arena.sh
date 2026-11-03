#!/bin/bash
# 82-desktop-arena.sh: idle-queue item (STAGED in ~/voxel-stage, not enqueued; 2026-11-03, mercury subagent for the GPU thread, gpu-scout-2026-11-03 pick 3).
# WHAT: three local models each drive the SAME Linux desktop (Xfce in the official hermes desktop sandbox, plus mousepad/thunar/gnumeric) through Hermes Agent `computer_use`
#   (cua-driver INSIDE a docker sandbox, terminal.backend docker), same 8 tasks, each scored by a script that reads the resulting files (gate outside the agent), one model resident at a
#   time, a per-task wall cap, and a split-screen clip built from three SEPARATE captures. Entrants: Qwen3.8-27B UD-Q4_K_M + mmproj-F16, Gemma-4-31B QAT q4_0 + mmproj, Ornith-1.5-35B Q4_K_M + mmproj.
# MOJO: three desktops side by side, one per local model, cursor-less windows opening and files appearing while a script marks each task PASS or FAIL | intense + never-seen (a desktop operated by a model at home through Hermes computer_use; checked 2026-11-03: no local-model comparative found in one search, never claimed as 'first') | line 1: "the smallest local model that can drive a Linux desktop in Hermes Agent is [X]: [n of 8] tasks at [N] GB and [t] tok/s"
# EV: line 1 above; rules: gpu-ev-gate-2026-10-25 section 2 rule 1 (which model, brand-pull entrants Qwen3.8 / Gemma 4 / Ornith) + rule 4 (fully local Hermes). comparables: 'computer use in Hermes Agent runs on Cua Driver' post 12,140 views, Hermes-on-local practical posts 7.9k-14.3k (scout section 3). bracket 4-14k. n = 1 attempt per model per task at sampling temperature, so a 'which model is best' claim needs the rerun it does not have: the post says 'cleared n of 8 in one pass each'.
# serves: content (a model verdict + a Hermes-native showcase) + hardening (is Hermes computer_use usable with local models on the pinned install: finds recorded in the status note).
#
# EVIDENCE BEHIND THE SETUP (all 2026-11-03, capsule CPU probes, vault 99-personal/benchmark-rig/desktop-arena-82-status.md):
#   - docs (live, fetched 2026-11-03): computer-use + bot-screen pages: with terminal.backend docker the driver and Xvnc+Xfce run INSIDE the sandbox; image nousresearch/hermes-sandbox:desktop.
#   - approvals: standard mode + command_allowlist cua:<action>:<background|foreground> grants (desk_mkhome.py), so every entrant drives the sandbox desktop
#     the docs describe, with the same pre-granted actions.
#   - tools.tool_search.enabled off (else computer_use sits behind tool_search/tool_describe/tool_call), toolset computer_use ONLY (no terminal/file tool to shortcut a task).
#   - input: on GTK apps the pin's cua-driver answers key presses with 'Background delivery is not available ... retry with delivery_mode:foreground'; the model has to follow the verdict.
#   - mock CPU proof: scripted model + real Hermes + real cua-driver + real Mousepad save, scored PASS (desk_runner.py --tasks t1, 10 s, 10 tool calls).
# SAMPLING per entrant = the maker card values already used in agenttest v2 rows.json (sampling_source there): Qwen 1.0/0.95/20/min-p 0.0, Gemma 1.0/0.95/64/0.05, Ornith 0.6/0.95/20/0.05.
# CONFOUNDS to label in every clip/number: Q4_K_M vs QAT q4_0 files; dense 27B / dense 31B / MoE 35B-A3B (speed is never a cross-model claim); three different mmproj files; thinking on with a 4096-token
#   budget for all (llama-server --reasoning-budget); one attempt per task; the derived desktop image d82-desktop:1 (official desktop sandbox + mousepad thunar gnumeric, built by this script if missing).
# resumable: needs >= 150 min left in the window (3 x (load 3 + 8 tasks x <=5 min cap + restarts) worst case ~145; expected ~95, pace UNVERIFIED until the smoke) or exits 3 (stays queued).
# artifact: $OUT/summary.json (exact). own-pid kills only (the server and Hermes children THIS item started), docker containers removed only by id from this item's own homes,
#   Donald drained with sudo -n systemctl stop, EXIT trap restores. Never :8090, never ~/idle-queue/ edits, no pkill by name.
set -uo pipefail
NL_TAG=desk82
MANIFEST=/home/witcheer/voxel-desk82-status.json
# throwaway manifest for a staging DRYRUN only (the drain greps the literal ^MANIFEST= line above)
[ -n "${MANIFEST_OVERRIDE:-}" ] && MANIFEST=$MANIFEST_OVERRIDE
export MANIFEST
. /home/witcheer/night-lib.sh
HD=${HD:-/home/witcheer/voxel-stage/desk82}            # harness dir (desk_*.py, image/Dockerfile)
V=/home/witcheer/voxelbench
OUT=${OUT:-$V/runs/desk82-arena}
BIN=/home/witcheer/llama.cpp-donald/build/bin/llama-server
GPY=/home/witcheer/benchmark-rig/.venv/bin/python
FFMPEG=$V/ffmpeg
HB=/home/witcheer/.hermes/hermes-agent/.hermes/bin/hermes
PIN=9bda1888bb61
IMG=d82-desktop:1
PORT=8283                       # our own llama-server port; never 8090 (Donald), 8190-8194/8196 are siblings
ARENA=${ARENA:-/tmp/arena82r-$(date +%s)}   # throwaway HERMES_HOMEs (short path, no account name); UNIQUE per run: docker-created subdirs are root-owned, so a stale home cannot be removed and a reused one carries a dead container pointer
CTX=131072; KV=q8_0; RB=4096; WALL=${WALL:-300}; FPS=3; NEED=${NEED:-150}
RB_MSG="Thinking budget reached. Act now: pick the next desktop action."
EXPECT_RUN=${EXPECT_RUN:-3}
# slug|caption|model file|model bytes|mmproj file|mmproj bytes|sampling flags|extra flags (space separated)|model GB label
ENTRANTS=(
"qwen38-27b-ud-q4km|Qwen3.8-27B UD-Q4_K_M|/home/witcheer/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf|16464440224|/home/witcheer/models/qwen38-27b/mmproj/mmproj-F16.gguf|927607488|--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0||16.5 GB + 0.9 GB mmproj"
"gemma4-31b-qat-q4_0|Gemma 4 31B QAT q4_0|/home/witcheer/models/gemma4-qat-q4/gemma-4-31B_q4_0-it.gguf|17650999456|/home/witcheer/models/gemma4-qat-q4/gemma-4-31B-it-mmproj.gguf|1200726016|--temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.05|--chat-template-kwargs {\"enable_thinking\":true}|17.7 GB + 1.2 GB mmproj"
"ornith15-35b-a3b-q4km|Ornith 1.5 35B-A3B Q4_K_M|/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf|21713462848|/home/witcheer/models/ornith-15-35b/mmproj/mmproj-Ornith-1.5-35B-BF16.gguf|902822240|--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05||21.7 GB + 0.9 GB mmproj"
)
# staging aid: ONLY=<slug glob> runs a subset of the entrants (a GPU smoke); the full item never sets it
if [ -n "${ONLY:-}" ]; then F=(); for r in "${ENTRANTS[@]}"; do case "${r%%|*}" in $ONLY) F+=("$r");; esac; done; ENTRANTS=("${F[@]}"); EXPECT_RUN=${#ENTRANTS[@]}; fi
ART=$OUT/summary.json

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}, ARENA_MOCK=${ARENA_MOCK:-none})"
nl_check_path "$HD/desk_lib.py" "$HD/desk_tasks.py" "$HD/desk_runner.py" "$HD/desk_mkhome.py" "$HD/desk_clip.py" "$HD/desk_mock.py" "$HD/image/Dockerfile" "$BIN" "$GPY" "$FFMPEG" "$HB"
"$GPY" -c "import PIL, ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$HD"/desk_*.py || nl_fail "harness syntax or PIL import"
# the pinned Hermes install the isolation recipe was proven on (never move it: other harnesses assert it)
HV=$($HB --version 2>&1 | head -1)
case "$HV" in *"$PIN"*) nl_say "  ok: Hermes pin $PIN ($HV)";; *) nl_fail "Hermes install is not the pin $PIN: $HV";; esac
HELP=$($BIN --help 2>&1 || true)
for f in --mmproj --jinja --reasoning-budget --reasoning-budget-message --chat-template-kwargs --cache-type-k --flash-attn --metrics --temp --top-p --top-k --min-p; do case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in this llama-server build";; esac; done
nl_say "  ok: llama-server flags exist (mmproj, jinja, reasoning-budget, sampling)"
# docker: the sandbox image (official desktop tag) and the derived arena image
docker image inspect nousresearch/hermes-sandbox:desktop >/dev/null 2>&1 && nl_say "  ok: base image present" || nl_fail "base image nousresearch/hermes-sandbox:desktop missing (docker pull, 1.2 GB)"
if docker image inspect "$IMG" >/dev/null 2>&1; then nl_say "  ok: arena image $IMG present"; else nl_say "  note: arena image $IMG missing, the RUN path builds it from $HD/image/Dockerfile (apt, ~1 min, needs the network)"; NEED_BUILD=1; fi
# port + sibling collisions
PU=$(ss -ltnH "sport = :$PORT"); [ -z "$PU" ] && nl_say "  ok: port $PORT free (never 8090)" || nl_fail "port $PORT is taken"
[ "$PORT" != "8090" ] || nl_fail "PORT must not be 8090"
NE=0
for row in "${ENTRANTS[@]}"; do
  IFS='|' read -r slug cap mfile mbytes pfile pbytes samp extra gb <<< "$row"
  NE=$((NE+1))
  [ -f "$mfile" ] && [ "$(stat -c %s "$mfile")" = "$mbytes" ] && nl_say "  ok: $slug model $(stat -c %s "$mfile") bytes" || nl_fail "$slug model file missing or size != $mbytes ($mfile)"
  [ -f "$pfile" ] && [ "$(stat -c %s "$pfile")" = "$pbytes" ] && nl_say "  ok: $slug mmproj $(stat -c %s "$pfile") bytes" || nl_fail "$slug mmproj missing or size != $pbytes ($pfile)"
  for f in $samp; do case "$f" in --*) case "$HELP" in *"$f"*) ;; *) nl_fail "flag $f not in build";; esac;; esac; done
done
nl_check_disk 120
# the ARENA_MOCK path proves orchestration CPU-only: no Donald drain, no llama-server, the scripted model on $PORT
if [ -n "${ARENA_MOCK:-}" ]; then [ -f "$ARENA_MOCK" ] || nl_fail "ARENA_MOCK script $ARENA_MOCK missing"; EXPECT_RUN=1; fi
if nl_skip_exact "$ART"; then nl_say "  SKIP: artifact exists: $ART"; PLAN_RUN=0; else nl_say "  RUN: $NE entrants x 8 tasks, artifact $ART"; PLAN_RUN=$NE; fi
[ -n "${ARENA_MOCK:-}" ] && PLAN_RUN=$((PLAN_RUN>0 ? 1 : 0))
[ "$PLAN_RUN" -eq "$EXPECT_RUN" ] || { nl_say "PLAN: planned RUN $PLAN_RUN != EXPECT_RUN $EXPECT_RUN (item complete or mismatch); nothing to do"; [ "$PLAN_RUN" = 0 ] && exit 0; exit 2; }
if [ -z "${ARENA_MOCK:-}" ]; then
  DL=$(nl_deadline_epoch); LEFT=$(( (DL-$(date +%s))/60 ))
  nl_say "  planned RUN: $NE entrants (<= $((8*WALL/60)) min of tasks each + load; hard stop $(date -d "@$DL" '+%a %H:%M') local, $LEFT min away)"
  [ "$LEFT" -ge "$NEED" ] || { nl_say "  under $NEED min left in the window: not starting (exit 3, resumable)"; exit 3; }
fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init arena; fi
nl_gate

# ---------------------------------------------------------------- RUN path
SRV=""; MOCKPID=""; HOMES=()
killsrv() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; SRV=""; }   # only the server THIS item started
cleanup_desktops() {   # stop each desktop through Hermes, then remove ONLY the container recorded in our own homes
  for h in "${HOMES[@]}"; do
    c=$("$GPY" -c "import json,sys;print(json.load(open(sys.argv[1]+'/bot-desktop/sandbox.json'))['container'])" "$h" 2>/dev/null)   # read BEFORE stop: stop removes the marker
    HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=$h timeout 90 $HB computer-use screen stop --force >/dev/null 2>&1
    [ -n "$c" ] && docker rm -f "$c" >/dev/null 2>&1
  done
}
restore() { cleanup_desktops; [ -n "$MOCKPID" ] && kill "$MOCKPID" 2>/dev/null; killsrv; sleep 3
  if [ -z "${ARENA_MOCK:-}" ] && [ -z "${DONALD_DONE:-}" ]; then nl_say "EXIT trap: restoring Donald"; nl_donald_restore; fi; }
trap restore EXIT
t0=$(date +%s)
mkdir -p "$OUT"
if [ "${NEED_BUILD:-0}" = "1" ]; then
  nl_say "building $IMG"
  ( cd "$HD/image" && docker build -t "$IMG" . > "$OUT/image-build.log" 2>&1 ) || { nl_mark arena failed "image build failed (see image-build.log)"; exit 1; }
fi
# scorers proven on a live desktop before the card is touched: fresh seed must FAIL, hand solution must PASS, all 8
PH=$("$GPY" "$HD/desk_mkhome.py" "$ARENA" prove 1 prove 65536 "$IMG") && HOMES+=("$PH")
( cd "$HD" && HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=$PH $HB computer-use screen start > "$OUT/prove-start.log" 2>&1 ) || { nl_mark arena failed "desktop would not start (prove)"; exit 1; }
( cd "$HD" && python3 desk_tasks.py --prove "$PH" > "$OUT/prove-scorers.txt" 2>&1 ) && nl_say "  scorer controls: $(grep -c '^OK' "$OUT/prove-scorers.txt") of 8 OK" || { nl_mark arena failed "scorer controls wrong: $(grep BAD "$OUT/prove-scorers.txt" | head -c 300)"; exit 1; }
cleanup_desktops; HOMES=()

start_server() {  # <slug> <model> <mmproj> <samp> <extra> <alias>  -> sets CTX_UP; 0 = up
  local slug=$1 model=$2 mmproj=$3 samp=$4 extra=$5 alias=$6 c up=0
  for c in $CTX 98304 65536; do
    # shellcheck disable=SC2086
    $BIN --model "$model" --mmproj "$mmproj" --host 127.0.0.1 --port $PORT --n-gpu-layers 99 --ctx-size $c --parallel 1 --flash-attn on --jinja --no-warmup \
      --cache-type-k $KV --cache-type-v $KV --alias "$alias" --metrics --reasoning-budget $RB --reasoning-budget-message "$RB_MSG" $samp $extra \
      > "$OUT/server-$slug.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 180); do hz=$(curl -s 127.0.0.1:$PORT/health); case "${hz,,}" in *ok*) up=1; break;; esac; kill -0 $SRV 2>/dev/null || break; sleep 2; done
    [ "$up" = 1 ] && { CTX_UP=$c; return 0; }
    nl_say "  server did not come up at ctx $c: $(grep -m1 -iE 'out of memory|failed|error|unknown' "$OUT/server-$slug.log" | head -c 200)"
    killsrv; sleep 4
  done
  return 1
}

if [ -z "${ARENA_MOCK:-}" ]; then
  nl_say "draining Donald"; nl_donald_stop
  for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
  [ "${v:-99999}" -lt 1024 ] || { nl_mark arena failed "Donald drain never cleared (${v} MiB), a foreign process may own the card"; exit 1; }
fi

N=0
for row in "${ENTRANTS[@]}"; do
  IFS='|' read -r slug cap mfile mbytes pfile pbytes samp extra gb <<< "$row"
  N=$((N+1)); alias=$(basename "$mfile")
  [ -n "${ARENA_MOCK:-}" ] && [ "$N" -gt 1 ] && break   # the mock proof is one pass
  [ -n "${ARENA_MOCK:-}" ] && { slug=mock; cap="mock model"; }
  EDIR=$OUT/$slug; mkdir -p "$EDIR"; nl_mark arena running "$slug"
  if [ -n "${ARENA_MOCK:-}" ]; then
    python3 "$HD/desk_mock.py" $PORT "$ARENA_MOCK" > "$EDIR/mock.log" 2>&1 &
    MOCKPID=$!; sleep 1; CTX_UP=65536
  else
    start_server "$slug" "$mfile" "$pfile" "$samp" "$extra" "$alias" || { nl_say "  $slug: server never came up"; echo "{\"label\": \"$slug\", \"error\": \"server_never_up\"}" >> "$OUT/entrant-errors.jsonl"; continue; }
    nl_say "server up $slug ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)), ctx $CTX_UP"
    ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> "$EDIR/vram-samples.csv"; sleep 20; done ) &
    VRAMLOG=$!
    curl -s 127.0.0.1:$PORT/props > "$EDIR/props.json"
    SCHK=$(python3 - "$EDIR/props.json" "$samp" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1])).get("default_generation_settings", {}); p = g.get("params", g)
fl = sys.argv[2].split(); want = {}
for k, name in (("--temp", "temperature"), ("--top-p", "top_p"), ("--top-k", "top_k"), ("--min-p", "min_p")):
    if k in fl: want[name] = float(fl[fl.index(k) + 1])
got = {k: p.get(k) for k in want}
ok = all(got[k] is not None and abs(float(got[k]) - v) < 1e-3 for k, v in want.items())
print(("OK " if ok else "BAD ") + json.dumps(got))
PYEOF
)
    nl_say "  server sampling defaults: $SCHK"
    case "$SCHK" in OK*) ;; *) nl_mark arena failed "$slug sampling defaults wrong: $SCHK"; kill "$VRAMLOG" 2>/dev/null; killsrv; continue;; esac
  fi
  # the entrant's own HERMES_HOME (never a profile of the main install), config rewritten with the context actually served
  H=$("$GPY" "$HD/desk_mkhome.py" "$ARENA" "$slug" $PORT "$alias" $CTX_UP "$IMG"); HOMES+=("$H")
  [ -n "${ARENA_MOCK:-}" ] && sed -i "s#^  default: .*#  default: mock#" "$H/config.yaml"
  ( cd "$HD" && python3 desk_runner.py --home "$H" --label "$slug" --out "$EDIR" --wall $WALL --fps $FPS ${TASKS:+--tasks $TASKS} ) >> "$EDIR/runner.log" 2>&1
  nl_say "  $slug runner rc=$? : $(grep -c . "$EDIR/results.jsonl" 2>/dev/null || echo 0) task rows, $(grep -c '"pass": true' "$EDIR/results.jsonl" 2>/dev/null || echo 0) passed"
  if [ -z "${ARENA_MOCK:-}" ]; then
    curl -s 127.0.0.1:$PORT/metrics | grep -E "^llamacpp:(tokens_predicted_total|prompt_tokens_total|tokens_predicted_seconds_total|prompt_seconds_total)" > "$EDIR/server-metrics.txt" 2>/dev/null
    kill "$VRAMLOG" 2>/dev/null; killsrv; sleep 3
    for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1024 ] && break; sleep 2; done
    [ "${v:-99999}" -lt 1024 ] || { nl_mark arena failed "card did not clear after $slug (${v} MiB)"; exit 1; }
  else
    kill "$MOCKPID" 2>/dev/null; MOCKPID=""
  fi
  cleanup_desktops; HOMES=()
done

# the card is free from here on: give Donald back BEFORE the CPU-only clip render (the EXIT trap then skips its restore)
if [ -z "${ARENA_MOCK:-}" ]; then nl_say "card released, restoring Donald before the CPU clip render"; nl_donald_restore; DONALD_DONE=1; fi
# summary (artifact) + clip, CPU only from here on
"$GPY" - "$OUT" <<'PYEOF'
import json, os, sys
out = sys.argv[1]; summ = {"entrants": {}, "tasks": 8, "wall_s": int(os.environ.get("WALL", "300"))}
for d in sorted(os.listdir(out)):
    rp = os.path.join(out, d, "results.jsonl")
    if not os.path.exists(rp): continue
    rows = [json.loads(l) for l in open(rp)]
    m = {}
    mp = os.path.join(out, d, "server-metrics.txt")
    if os.path.exists(mp):
        for l in open(mp):
            k, v = l.split(); m[k.split(":")[1]] = float(v)
    tps = round(m["tokens_predicted_total"] / m["tokens_predicted_seconds_total"], 1) if m.get("tokens_predicted_seconds_total") else None
    summ["entrants"][d] = {"passed": sum(1 for r in rows if r["pass"]), "n": len(rows), "per_task": {r["task"]: ("timeout" if r["timeout"] else ("pass" if r["pass"] else "fail")) for r in rows},
                           "tool_calls": sum(r["tool_calls"] for r in rows), "tool_errors": sum(r["tool_errors"] for r in rows), "secs": round(sum(r["secs"] for r in rows)), "decode_tok_s": tps}
json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(summ))
PYEOF
if nl_judge_artifact arena 0 "$ART" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then
  nl_say "  summary: $(tr -d '\n ' < "$ART" | head -c 600)"
  CAPS=(); for row in "${ENTRANTS[@]}"; do IFS='|' read -r slug cap mfile mbytes pfile pbytes samp extra gb <<< "$row"; [ -n "${ARENA_MOCK:-}" ] && { slug=mock; cap="mock model"; gb="scripted"; }; [ -d "$OUT/$slug" ] && CAPS+=(--entrant "$slug=$OUT/$slug=$cap|$gb"); done
  setsid timeout 900 nice -n 19 "$GPY" "$HD/desk_clip.py" --out "$OUT/desktop-arena.mp4" --ffmpeg "$FFMPEG" --still "$OUT/still.png" "${CAPS[@]}" < /dev/null >> "$OUT/clip.log" 2>&1 \
    && nl_say "  clip: $OUT/desktop-arena.mp4 (read $OUT/still.png and 3 sampled frames by eye before any post: only the sandbox clock and window titles are in the panes)" || nl_say "  clip render failed (frames + results stand)"
fi
nl_say "DONE desk82"
exit 0
