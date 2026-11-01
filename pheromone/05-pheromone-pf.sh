#!/bin/bash
# 05-pheromone-pf.sh: idle-queue item, esoteric lane "pheromone harness" legs 1+2. STAGED 2026-10-30 by mercury.
# Provenance: operator in the esoteric thread 2026-10-30 ~13:50 CET, "yes perfect! doing A" = option A: runs 1 and 2 on Saturday, run 3 (Hermes) on Sunday.
#   vault status: 99-personal/esoteric/fit-check-2601-08129.md. Booked block: Saturday 2026-10-31 only (window guard below); outside it the item exits 3 in 0 s.
# EV: protects the leg-3 post (4 Hermes Agent profiles on local weights coordinating only through a fading marks file, filmed as a heatmap: gate rule 4 fully local Hermes piece + a picture)
#   and carries the esoteric hook on its own if it holds: a Jan 2026 arXiv claim (2601.08129: 48.5% vs 11.1% conversation) re-run on our card, then the question the paper skipped
#   (a capable 2026 model, 1 vs 4 agents, a token-matched conversation arm). projected line 1 either "four agents that never talk did no better than one" or
#   "agents that never talk beat agents that do, even when the talkers get the same token budget". a table-only readout is hardening for the leg-3 post, not a post.
# serves: experiment (method question: is the win stigmergy, or a validated search loop?) + content (protects the leg-3 Hermes piece)
# What it measures: solve rate on the paper's meeting-room scheduling task (Govcraft/pressure-field-experiment @ d3b0d5b, built unchanged; Rust), one grid call per cell, own json per cell.
#   arm "paper": the paper's models qwen2.5:0.5b -> 1.5b -> 3b escalation chain on Ollama (the paper's own runtime, Ollama 0.24 client here; solve rates only, no speed claims).
#   arm "moe": Ornith 1.5 35B-A3B Q4_K_M on llama.cpp (bolt build cd0a5b6a1), thinking off (--reasoning-budget 0), alias "ornith-moe" (an alias holding 0.5b/1.5b/3b/7b/14b would be
#     routed to a hard-coded vLLM port by vllm_client.rs model_to_port), chain = the same alias three times so both strategies keep the paper's 50-tick budget (with a one-model chain the
#     pressure-field run ends at tick 24 "converged at largest model and highest band" while conversation runs to 50: smoke 2026-10-30).
# Deviations (all logged into run-config.json): RTX 5090 instead of a laptop RTX 4070 8GB; decay/inhibition are wall-clock (fitness half-life 5 s, 2 s cooldown), so a faster card
#   decays less per tick; trials per cell 30 (easy, paper arm) / 20 / 12 (moe medium) instead of 30 everywhere; agents 1 and 4 only; hierarchical/sequential/random not re-run;
#   the decay ablation uses a second build (~/esoteric/pf-ablate2, a worktree of the same commit) whose only change is the ablation config list cut to full + no_decay (7 deleted lines).
# What it does NOT measure: speed, hard tier unless time is left, any Hermes Agent behaviour (that is leg 3).
# resumable per cell: <cell>.done marker (exact path). a cell killed at the window end keeps its incremental json and is marked partial (not re-run). exit 3 while cells remain.
# test knobs (empty in production): PH_RES, PH_CELLS (space-separated ids), PH_TRIALS (override every cell's trials), PH_WIN_START / PH_WIN_END (epochs), PH_MANIFEST, PH_RUNLOG.
set -uo pipefail
NL_TAG=pheromone
MANIFEST=${PH_MANIFEST:-/home/witcheer/esoteric/pheromone-status.json}
export MANIFEST
. /home/witcheer/night-lib.sh
[ "${DRYRUN:-0}" = "1" ] && [ -z "${PH_MANIFEST:-}" ] && MANIFEST=/tmp/pheromone-dryrun.json
E=/home/witcheer/esoteric
BIN=$E/pressure-field/target/release/schedule-experiment
BIN2=$E/pf-ablate2/target/release/schedule-experiment
SUM=$E/pheromone_summary.py
SRV=/home/witcheer/llama.cpp-bolt/build/bin/llama-server
MODEL=/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf
OLL=/usr/local/bin/ollama
OMODELS=$E/ollama-models
RES=${PH_RES:-$E/pheromone-results}
RUNLOG=${PH_RUNLOG:-$E/pheromone-run.log}
PAPER_CHAIN=qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b
MOE_CHAIN=ornith-moe,ornith-moe,ornith-moe
OH=http://127.0.0.1:11434
MP=8174; MH=http://127.0.0.1:$MP
WIN_START=${PH_WIN_START:-$(date -d '2026-10-31 00:05' +%s)}
WIN_END=${PH_WIN_END:-$(date -d '2026-10-31 22:00' +%s)}
mkdir -p "$RES"

# cell table: id | binary (grid|ablate) | arm (paper|moe) | args (trials last as -t N; @MATCH:<pf>:<cv> resolved at run time)
CELLS_ALL=(
 "pe-pf4|grid|paper|-s pressure_field -a 4 -d easy -t 30"
 "pe-cv4|grid|paper|-s conversation -a 4 -d easy -t 30"
 "me-pf4|grid|moe|-s pressure_field -a 4 -d easy -t 20"
 "me-cv4|grid|moe|-s conversation -a 4 -d easy -t 20"
 "pe-pf1|grid|paper|-s pressure_field -a 1 -d easy -t 30"
 "me-pf1|grid|moe|-s pressure_field -a 1 -d easy -t 20"
 "pe-decay|ablate|paper|-a 4 -d easy -t 30"
 "me-cvM|grid|moe|-s conversation -a 4 -d easy --max-ticks @MATCH:me-pf4:me-cv4 -t 20"
 "pm-pf4|grid|paper|-s pressure_field -a 4 -d medium -t 20"
 "pm-cv4|grid|paper|-s conversation -a 4 -d medium -t 20"
 "mm-pf4|grid|moe|-s pressure_field -a 4 -d medium -t 12"
 "mm-cv4|grid|moe|-s conversation -a 4 -d medium -t 12"
 "mm-cvM|grid|moe|-s conversation -a 4 -d medium --max-ticks @MATCH:mm-pf4:mm-cv4 -t 12"
 "me-decay|ablate|moe|-a 4 -d easy -t 20"
 "ph-pf4|grid|paper|-s pressure_field -a 4 -d hard -t 20"
 "ph-cv4|grid|paper|-s conversation -a 4 -d hard -t 20"
)
CELLS=()
for c in "${CELLS_ALL[@]}"; do
  id=${c%%|*}
  if [ -n "${PH_CELLS:-}" ]; then case " $PH_CELLS " in *" $id "*) ;; *) continue;; esac; fi
  CELLS+=("$c")
done

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
NOW=$(nl_hc_now)
if [ "${DRYRUN:-0}" != "1" ] && { [ "$NOW" -lt "$WIN_START" ] || [ "$NOW" -ge "$WIN_END" ]; }; then
  nl_say "window guard: operator booked Sat 2026-10-31 00:05-22:00 local ($(date -d @$WIN_START '+%a %H:%M')-$(date -d @$WIN_END '+%a %H:%M')), not now (exit 3, resumable)"; exit 3
fi
nl_check_path "$BIN" "$BIN2" "$SUM" "$SRV" "$MODEL" "$OLL"
for m in 0.5b 1.5b 3b; do nl_check_path "$OMODELS/manifests/registry.ollama.ai/library/qwen2.5/$m"; done
HELP=$("$BIN" --help 2>&1 || true); GH=$("$BIN" grid --help 2>&1 || true); AH=$("$BIN2" ablation --help 2>&1 || true)
for f in --host --model-chain; do case "$HELP" in *"$f"*) ;; *) nl_fail "harness flag $f missing";; esac; done
for f in --strategies --agents --difficulties --trials --max-ticks --output; do case "$GH" in *"$f"*) ;; *) nl_fail "grid flag $f missing";; esac; done
for f in --agents --difficulty --trials --output; do case "$AH" in *"$f"*) ;; *) nl_fail "ablation flag $f missing";; esac; done
SH=$("$SRV" --help 2>&1 || true); for f in --reasoning-budget --parallel --alias --jinja; do case "$SH" in *"$f"*) ;; *) nl_fail "llama-server flag $f missing";; esac; done
HEAD1=$(git -C $E/pressure-field rev-parse --short HEAD 2>/dev/null); [ "$HEAD1" = "d3b0d5b" ] || nl_fail "harness HEAD $HEAD1 != d3b0d5b"
[ -z "$(git -C $E/pressure-field status --porcelain --untracked-files=no 2>/dev/null)" ] || nl_fail "harness clone has local changes"
DST=$(git -C $E/pf-ablate2 diff --shortstat 2>/dev/null); case "$DST" in *"1 file changed, 7 deletions(-)"*) nl_ok "ablate2 diff: $DST";; *) nl_fail "ablate2 diff unexpected: '$DST'";; esac
nl_check_disk 30
PHASES=(); NRUN=0
for c in "${CELLS[@]}"; do
  IFS='|' read -r id kind arm args <<<"$c"; PHASES+=("$id")
  if nl_skip_exact "$RES/$id.done"; then nl_say "  SKIP $id ($RES/$id.done)"; else nl_say "  RUN  $id [$kind/$arm] $args"; NRUN=$((NRUN+1)); fi
done
nl_say "  planned RUN: $NRUN of ${#CELLS[@]} cells; window ends $(date -d @$WIN_END '+%a %H:%M') local"
if [ "$NRUN" = 0 ]; then nl_say "nothing to run, item complete"; exit 0; fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init "${PHASES[@]}"; fi
nl_gate

OPID=""; MPID=""
restore() {
  [ -n "$OPID" ] && pkill -TERM -P "$OPID" 2>/dev/null   # ollama model runners are children of the serve pid
  [ -n "$MPID" ] && kill "$MPID" 2>/dev/null; [ -n "$OPID" ] && kill "$OPID" 2>/dev/null; sleep 5
  [ -n "$OPID" ] && pkill -9 -P "$OPID" 2>/dev/null
  [ -n "$MPID" ] && kill -9 "$MPID" 2>/dev/null; [ -n "$OPID" ] && kill -9 "$OPID" 2>/dev/null; sleep 2
  nl_say "EXIT trap: restoring Donald"; nl_donald_restore
}
trap restore EXIT
nl_say "draining Donald"; nl_donald_stop
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1500 ] && { nl_say "drain clear (${v} MiB)"; break; }; sleep 2; done
[ "${v:-99999}" -lt 1500 ] || { nl_say "card never drained (${v} MiB): abort"; exit 1; }

OLLAMA_MODELS=$OMODELS OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=30m "$OLL" serve >> "$E/ollama-serve.log" 2>&1 & OPID=$!
"$SRV" -m "$MODEL" --alias ornith-moe --host 127.0.0.1 --port $MP -ngl 99 -c 32768 --parallel 4 --jinja --reasoning-budget 0 -fa on >> "$E/pheromone-llama.log" 2>&1 & MPID=$!
UP=0; for i in $(seq 1 90); do curl -s $MH/health | grep -qi ok && curl -s $OH/api/version | grep -q version && { UP=1; break; }; sleep 2; done
[ "$UP" = 1 ] || { nl_say "servers did not come up (llama $(curl -s $MH/health), ollama $(curl -s $OH/api/version))"; exit 1; }
OVER=$(curl -s $OH/api/version); LVER=$("$SRV" --version 2>&1 | grep -m1 -i version | tr -d '"')
curl -s $MH/props > "$RES/moe-props.json" 2>/dev/null
python3 - "$RES/run-config.json" "$OVER" "$LVER" "$HEAD1" "$DST" <<'PY'
import json, sys, time, subprocess
p, over, lver, head, dst = sys.argv[1:6]
try: cfg = json.load(open(p))
except Exception: cfg = {"starts": []}
cfg.setdefault("starts", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ollama": over, "llama_cpp": lver,
  "harness": "Govcraft/pressure-field-experiment@" + head, "ablate2_diff": dst,
  "paper_chain": "qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b (Ollama library tags, default quant)",
  "moe": "Ornith-1.5-35B-Q4_K_M.gguf, llama-server -c 32768 --parallel 4 --jinja --reasoning-budget 0 -fa on, harness-sent temperature/top_p (sampling bands), server defaults otherwise",
  "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()})
json.dump(cfg, open(p, "w"), indent=1)
PY
nl_say "servers up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); ollama $OVER; $LVER"

REMAIN=0
for c in "${CELLS[@]}"; do
  IFS='|' read -r id kind arm args <<<"$c"
  nl_skip_exact "$RES/$id.done" && continue
  NOW=$(date +%s)
  if [ "$NOW" -ge $((WIN_END - 900)) ]; then nl_say "  $id: under 15 min left in the booked window, leaving it queued"; REMAIN=$((REMAIN+1)); continue; fi
  if [[ "$args" == *@MATCH:* ]]; then
    tag=$(grep -o '@MATCH:[^ ]*' <<<"$args"); IFS=':' read -r _ pfc cvc <<<"$tag"
    if [ ! -f "$RES/$pfc.done" ] || [ ! -f "$RES/$cvc.done" ]; then nl_say "  $id: WAIT ($pfc / $cvc not done)"; REMAIN=$((REMAIN+1)); continue; fi
    mt=$(python3 "$SUM" "$RES" --match "$pfc" "$cvc") || { nl_mark "$id" failed "match calc"; REMAIN=$((REMAIN+1)); continue; }
    args=${args/$tag/$mt}; nl_say "  $id: token-matched conversation budget = $mt ticks (from $pfc vs $cvc)"
  fi
  if [ -n "${PH_TRIALS:-}" ]; then args=$(sed -E "s/-t [0-9]+/-t $PH_TRIALS/" <<<"$args"); fi
  host=$OH; chain=$PAPER_CHAIN; [ "$arm" = moe ] && { host=$MH; chain=$MOE_CHAIN; }
  b=$BIN; sub=grid; [ "$kind" = ablate ] && { b=$BIN2; sub=ablation; }
  left=$(( WIN_END + 1800 - NOW ))
  t0=$(date +%s); nl_mark "$id" running "$args"; nl_say "  $id: start ($sub $args, host $host, timeout ${left}s)"
  # shellcheck disable=SC2086
  ( cd $E/pressure-field && timeout "$left" "$b" --host "$host" --model-chain "$chain" $sub $args -o "$RES/$id.json" ) >> "$RUNLOG" 2>&1; rc=$?
  dur=$(( ($(date +%s)-t0)/60 ))
  if [ "$rc" = 0 ] && [ -s "$RES/$id.json" ]; then
    touch "$RES/$id.done"; nl_judge_artifact "$id" 0 "$RES/$id.done" "$t0" "$dur min"
    nl_say "  $id: done in $dur min: $(python3 "$SUM" "$RES" 2>/dev/null | grep "^$id" | tr '\n' ';')"
  elif [ "$rc" = 124 ] && [ -s "$RES/$id.json" ]; then
    echo "partial: killed at window end after $dur min" > "$RES/$id.done"; nl_mark "$id" done "partial at window end, $dur min"
    nl_say "  $id: PARTIAL (window end) after $dur min, kept"
  else
    nl_mark "$id" failed "rc=$rc after $dur min"; nl_say "  $id: FAILED rc=$rc after $dur min (see $RUNLOG)"; REMAIN=$((REMAIN+1))
  fi
done
python3 "$SUM" "$RES" > /dev/null 2>&1 && nl_say "summary: $RES/summary.md"
if [ "$REMAIN" -gt 0 ]; then nl_say "DONE rc=0 ($REMAIN cells still queued/resumable)"; exit 3; fi
nl_say "DONE rc=0 (all cells done)"; exit 0
