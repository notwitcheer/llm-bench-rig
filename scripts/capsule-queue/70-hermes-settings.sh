#!/bin/bash
# 70-hermes-settings.sh: idle-queue item, STAGED 2026-10-26 by mercury (subagent, GPU thread). Sorts first among bench items (70-); duck items (09*) yield or run by the weekday alternation.
# EV: private (projected post angle and reach bracket live in the ops vault status note)
# serves: content (reproducible fully-local Hermes piece) + hardening (run-to-run noise of the frozen toolcall-v1 slice, paired exact McNemar against the 10-24 baseline rows)
# What it measures: Qwen3.8-27B Q6_K, the toolcall-v1 baseline row (mainline b10371 5d16e81dd, board server line `--n-gpu-layers 99 --jinja --ctx-size N`, raw /completion, official template client-side,
#   greedy, parser toolcall-v1-parser-2, frozen list FROZEN.sha256, max_tokens 16384, ctx 24576 single-turn / 40960 S5, server port 8097). One factor per leg against the existing rows
#   results/toolcall-v1/qwen3-8-27b-q6-k-{st-on,st-off,s5-on} (st-on 799/906 scored, st-off 784, s5-on 81/100, all KV f16, think per leg name). Results: ~/benchmark-rig/results/hermes-settings/<leg>/.
# What it does NOT measure: the context legs run the unchanged slice (S5 peaks at 14,951 prompt+gen tokens), so a 32k/64k/128k allocation is expected to change nothing in the outputs; that is what the
#   paired identity check proves or refutes. What 128k really costs (VRAM, prefill/decode tok/s at depth, a passkey at 120k) is the `fit` leg (3 KV types x 3 ctx, deep sweep at 131072).
#   Donald's own flags (llama.cpp-donald + MTP draft, q8_0 KV, ctx 65536, parallel 1, flash-attn on) are NOT this regime; the board binary and server line are used so the baseline stays comparable.
# legs, cheapest/most postable first (each resumable on exact-path artifacts, hard stop = end of the heures creuses block, exit 3 while legs remain; est = card minutes at toolcall-v1 measured pace):
#   fit              ~14  KV f16/q8_0/q4_0 x ctx 32768/65536/131072: VRAM, load, prefill+decode tok/s at depth 0..120k, passkey (131072 cells deep, others one request). smoke 2026-10-26 08:10: f16 131072 loads at 30,104 MiB (fits 32 GB, 6 deep requests 189 s), q8_0 26,734, q4_0 24,686
#   s5-off-f16       ~28  S5, think OFF, f16, ctx 40960          (thinking post: with the existing st-on 799 / st-off 784 and s5-on 81)
#   s5-on-q8         ~61  S5, think on, K+V q8_0, ctx 40960      (KV post, multi-turn half)
#   s5-on-f16-r2     ~61  S5, think on, f16, ctx 40960, EXACT repeat of the 10-24 baseline (run-to-run noise / determinism)
#   st-on-q8         ~79  S1..S4, think on, K+V q8_0, ctx 24576  (KV post, single-turn half)
#   st-on-f16-r2     ~79  S1..S4 exact repeat of the baseline (noise)
#   s5-on-f16-c32k   ~61  S5, f16, ctx 32768                     (context post: identity vs baseline at 40960)
#   s5-on-f16-c64k   ~61  S5, f16, ctx 65536
#   s5-on-f16-c128k  ~61  S5, f16, ctx 131072 (fit proven by the fit leg; a load OOM writes load-failed.json and the leg counts as done-NOFIT)
#   s5-on-q4         ~61  S5, think on, K+V q4_0, ctx 40960      (only if the q8 result makes q4 interesting; last on purpose)
#   total ~566 card min + ~10 server loads (smoke 2026-10-26: S5 think-off 3 entries 59 s, q8_0 server loads and answers, paired reader and --ids knob verified). After the legs: paired readout (CPU) -> results/hermes-settings/paired-readout.txt (exact McNemar + identical-output rate per pair).
set -uo pipefail
NL_TAG=hset
MANIFEST=/home/witcheer/hermes-settings-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
[ "${DRYRUN:-0}" = "1" ] && MANIFEST=/tmp/hermes-settings-dryrun.json   # a DRYRUN must never wipe the real manifest (the drain DRYRUNs before every block)
# smoke-test knobs (all empty in production): HSET_RES results dir, HSET_MANIFEST, HSET_RUNLOG, HSET_LEGS (space-separated leg names to plan), HSET_FITCELLS (kv:ctx:plan ...), HSET_IDS (driver --ids json instead of --sets)
[ -n "${HSET_MANIFEST:-}" ] && MANIFEST=$HSET_MANIFEST
T=/home/witcheer/toolcall-v1
PY=$T/.venv/bin/python
RES=${HSET_RES:-/home/witcheer/benchmark-rig/results/hermes-settings}
BASE=/home/witcheer/benchmark-rig/results/toolcall-v1
RUNLOG=${HSET_RUNLOG:-/home/witcheer/hermes-settings-run.log}
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
Q6K=/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
PORT=8097
BASE_ST_ON=$BASE/qwen3-8-27b-q6-k-st-on
BASE_ST_OFF=$BASE/qwen3-8-27b-q6-k-st-off
BASE_S5_ON=$BASE/qwen3-8-27b-q6-k-s5-on
# name|est_min|kind|kv|ctx|think|sets|baseline dir for the per-leg paired line
LEGS=(
  "fit|14|fit|all|131072|off|-|-"
  "s5-off-f16|28|drv|f16|40960|off|S5|$BASE_S5_ON"
  "s5-on-q8|61|drv|q8_0|40960|on|S5|$BASE_S5_ON"
  "s5-on-f16-r2|61|drv|f16|40960|on|S5|$BASE_S5_ON"
  "st-on-q8|79|drv|q8_0|24576|on|S1,S2,S3,S4|$BASE_ST_ON"
  "st-on-f16-r2|79|drv|f16|24576|on|S1,S2,S3,S4|$BASE_ST_ON"
  "s5-on-f16-c32k|61|drv|f16|32768|on|S5|$BASE_S5_ON"
  "s5-on-f16-c64k|61|drv|f16|65536|on|S5|$BASE_S5_ON"
  "s5-on-f16-c128k|61|drv|f16|131072|on|S5|$BASE_S5_ON"
  "s5-on-q4|61|drv|q4_0|40960|on|S5|$BASE_S5_ON"
)
FITCELLS=( "f16:131072:deep" "q8_0:131072:deep" "q4_0:131072:deep" "f16:65536:vram" "q8_0:65536:vram" "q4_0:65536:vram" "f16:32768:vram" "q8_0:32768:vram" "q4_0:32768:vram" )
[ -n "${HSET_FITCELLS:-}" ] && read -r -a FITCELLS <<< "$HSET_FITCELLS"
if [ -n "${HSET_LEGS:-}" ]; then KEEP=(); for L in "${LEGS[@]}"; do n=${L%%|*}; case " $HSET_LEGS " in *" $n "*) KEEP+=("$L");; esac; done; LEGS=("${KEEP[@]}"); fi
art_of() { case "$1" in fit) echo "$RES/fit/summary.json";; *) echo "$RES/$1/summary.json";; esac; }
nofit_of() { echo "$RES/$1/load-failed.json"; }
done_leg() { [ -f "$(art_of "$1")" ] || [ -f "$(nofit_of "$1")" ]; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$MAIN" "$Q6K" "$PY" "$T/code/driver.py" "$T/code/hermes_settings_fitdepth.py" "$T/code/hermes_settings_paired.py" \
  "$T/frozen/toolcall-v1/manifest.json" "$T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz" "$T/qwen38-tokenizer/tokenizer.json" \
  "$BASE_ST_ON/summary.json" "$BASE_ST_OFF/summary.json" "$BASE_S5_ON/summary.json"
nl_check_disk 20
( cd $T/frozen/toolcall-v1 && sha256sum -c --quiet FROZEN.sha256 ) && nl_ok "frozen list sha256 intact" || nl_fail "frozen list changed"
( cd $T/code && $PY -c "import qwen38_xml as X; assert X.PARSER_VERSION == 'toolcall-v1-parser-2', X.PARSER_VERSION" 2>/dev/null ) && nl_ok "parser toolcall-v1-parser-2" || nl_fail "parser version changed (re-run roundtrip_test.py and update this item)"
case "$($MAIN --version 2>&1)" in *5d16e81*) nl_ok "mainline 5d16e81 (b10371)";; *) nl_fail "mainline build changed";; esac
HELP=$($MAIN --help 2>&1 || true)
for want in "--cache-type-k" "--cache-type-v" "--ctx-size" "--flash-attn"; do case "$HELP" in *"$want"*) :;; *) nl_fail "binary lacks $want";; esac; done
nl_ok "binary lists --cache-type-k/-v, --ctx-size, --flash-attn"
ST=$(wc -l < "$BASE_ST_ON/items.jsonl"); [ "$ST" -ge 899 ] && nl_ok "baseline st-on items: $ST" || nl_fail "baseline st-on incomplete ($ST)"
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$T/code/hermes_settings_fitdepth.py" "$T/code/hermes_settings_paired.py" && nl_ok "helper scripts parse" || nl_fail "helper scripts do not parse"
RUNS=0; EST=0; PH=""
for L in "${LEGS[@]}"; do IFS='|' read -r name est kind kv ctx think sets basedir <<< "$L"
  PH="$PH $name"
  if done_leg "$name"; then nl_say "  $name: SKIP, artifact present ($( [ -f "$(art_of "$name")" ] && art_of "$name" || nofit_of "$name"))"
  else nl_say "  $name: RUN  (kv $kv ctx $ctx think $think sets $sets, est ${est} min; checks $(art_of "$name"))"; RUNS=$((RUNS+1)); EST=$((EST+est)); fi
done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS legs, est $EST card min; hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away"
[ -z "${EXPECT_RUN:-}" ] || [ "$RUNS" = "$EXPECT_RUN" ] || { nl_say "ABORT: planned RUN $RUNS != EXPECT_RUN $EXPECT_RUN"; exit 2; }
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

SP=""
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_vram_free() { local i v; for i in $(seq 1 60); do v=$(vram); [ "${v:-99999}" -lt 1500 ] && return 0; sleep 2; done; return 1; }
stop_server() { # own pid only, never pkill: a sibling's llama-server must survive
  [ -n "$SP" ] && kill "$SP" 2>/dev/null; sleep 3; [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && { kill -9 "$SP" 2>/dev/null; sleep 2; }
  SP=""; wait_vram_free || nl_say "  WARN: VRAM still $(vram) MiB after server stop"; }
restore() { stop_server; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
  for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:8090/health | grep -qi ok && { nl_say "Donald proxy health OK"; break; }; sleep 2; done; }
trap restore EXIT
kvflags() { case "$1" in f16) echo "";; q8_0) echo "--cache-type-k q8_0 --cache-type-v q8_0";; q4_0) echo "--cache-type-k q4_0 --cache-type-v q4_0";; esac; }
start_server() { # <kv> <ctx> <server.log>  -> 0 up, 1 failed, 4 failed with an out-of-memory signature
  local kv=$1 ctx=$2 log=$3
  # shellcheck disable=SC2046
  "$MAIN" -m "$Q6K" --port $PORT --n-gpu-layers 99 --host 127.0.0.1 --jinja --ctx-size "$ctx" $(kvflags "$kv") >> "$log" 2>&1 &
  SP=$!
  local i; for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && return 0
  local oom=1; tail -200 "$log" | grep -qiE "out of memory|failed to allocate|cudaMalloc failed|ggml_backend_cuda_buffer_type_alloc_buffer" && oom=4
  stop_server; return $oom; }

nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(vram); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
mkdir -p "$RES"
DEADLINE=$(nl_deadline_epoch)

run_fit() { # leg "fit": every (kv, ctx) cell on its own server; a cell json per cell is the resume unit
  local name=fit d="$RES/fit" cell kv ctx plan left rc t0 ncell=0 nfail=0
  mkdir -p "$d"
  for cell in "${FITCELLS[@]}"; do IFS=: read -r kv ctx plan <<< "$cell"
    local cj="$d/${kv}-c${ctx}.json"
    [ -f "$cj" ] && { ncell=$((ncell+1)); continue; }
    left=$(( DEADLINE - $(date +%s) - 120 ))
    [ "$left" -gt 420 ] || { nl_say "  fit: window exhausted before $kv ctx $ctx ($left s left), resumable"; return 3; }
    nl_say "  fit cell $kv ctx $ctx ($plan)"
    start_server "$kv" "$ctx" "$d/${kv}-c${ctx}.server.log"; rc=$?
    if [ "$rc" = 4 ]; then
      printf '{"label":"%s-c%s","kv":"%s","ctx":%s,"load_failed":true,"reason":"out-of-memory signature in server log","vram_mib_now":%s,"log_tail":%s}\n' "$kv" "$ctx" "$kv" "$ctx" "$(vram)" \
        "$(tail -8 "$d/${kv}-c${ctx}.server.log" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-900:]))')" > "$cj"
      nl_say "  fit $kv ctx $ctx: DOES NOT FIT (out of memory at load), recorded"; ncell=$((ncell+1)); continue
    elif [ "$rc" != 0 ]; then nl_say "  fit $kv ctx $ctx: server did not come up (not an OOM signature, resumable)"; nfail=$((nfail+1)); continue; fi
    nl_say "  server up, VRAM $(vram) MiB"
    t0=$(date +%s)
    timeout $(( left + 30 )) $PY -u $T/code/hermes_settings_fitdepth.py --server http://127.0.0.1:$PORT --label "${kv}-c${ctx}" --out "$cj" --kv "$kv" --ctx "$ctx" --plan "$plan" >> "$d/${kv}-c${ctx}.driver.log" 2>&1; rc=$?
    stop_server
    if [ "$rc" = 0 ] && [ -f "$cj" ]; then nl_say "  fit $kv ctx $ctx done in $(( $(date +%s)-t0 )) s: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('loaded %s MiB, peak seen %s MiB, ' % (r['vram_mib_loaded'], r['vram_mib_peak_seen']) + ' '.join('%sk:%s/%s tok/s%s' % (q['target_tokens']//1000, q.get('prefill_tok_s'), q.get('decode_tok_s'), '' if q.get('found') else ' NOKEY') for q in r['requests'] if 'prefill_tok_s' in q))" "$cj" 2>/dev/null)"; ncell=$((ncell+1))
    else nl_say "  fit $kv ctx $ctx FAILED rc=$rc (resumable)"; rm -f "$cj"; nfail=$((nfail+1)); fi
  done
  if [ "$ncell" = "${#FITCELLS[@]}" ]; then
    python3 - "$d" <<'PY' && return 0
import glob, json, sys
d = sys.argv[1]
cells = [json.load(open(p)) for p in sorted(glob.glob(d + "/*.json")) if not p.endswith("summary.json")]
json.dump({"cells": cells, "n": len(cells)}, open(d + "/summary.json", "w"), indent=1)
print("[hset] fit summary: %d cells" % len(cells))
PY
  fi
  return 3
}

run_drv() { # <name> <kv> <ctx> <think> <sets> <basedir>
  local name=$1 kv=$2 ctx=$3 think=$4 sets=$5 basedir=$6 d="$RES/$1" left t0 rc blabel gp selarg
  left=$(( DEADLINE - $(date +%s) - 120 ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $name ($left s left), resumable"; nl_mark "$name" failed "window exhausted (resumable)"; return 3; }
  blabel="ggml-org/llama.cpp mainline b10371 (5d16e81dd); KV cache K=$kv V=$kv; flash-attn auto"
  mkdir -p "$d"; t0=$(date +%s); nl_mark "$name" running; nl_say "=== $name (kv $kv, ctx $ctx, think $think, $sets, budget $((left/60)) min) ==="
  start_server "$kv" "$ctx" "$d/server.log"; rc=$?
  if [ "$rc" = 4 ]; then
    printf '{"label":"%s","kv":"%s","ctx":%s,"load_failed":true,"reason":"out-of-memory signature in server log","vram_mib_now":%s}\n' "$name" "$kv" "$ctx" "$(vram)" > "$(nofit_of "$name")"
    nl_say "  $name: DOES NOT FIT at ctx $ctx with KV $kv (out of memory at load), recorded as NOFIT"; nl_mark "$name" skipped "NOFIT"; return 0
  elif [ "$rc" != 0 ]; then nl_say "  $name: server did not come up (not an OOM signature, resumable)"; nl_mark "$name" failed "server load"; return 3; fi
  nl_say "  server up, VRAM $(vram) MiB"
  nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader -l 10 > "$d/gpu-samples.csv" 2>/dev/null &
  gp=$!   # the nvidia-smi pid itself (a subshell wrapper would leave the loop orphaned)
  if [ -n "${HSET_IDS:-}" ]; then selarg="--ids $HSET_IDS"; else selarg="--sets $sets"; fi
  ( cd $T/code && timeout $(( left + 60 )) $PY -u driver.py --server http://127.0.0.1:$PORT --label "hset-$name" $selarg \
      --manifest $T/frozen/toolcall-v1/manifest.json --tokenizer $T/qwen38-tokenizer --prompts $T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz \
      --out "$d" --think "$think" --max-tokens 16384 --ctx "$ctx" --budget-s "$left" --tokenize-all --server-bin "$blabel" >> "$RUNLOG" 2>&1 )
  rc=$?
  kill $gp 2>/dev/null; stop_server
  if [ "$rc" = 3 ]; then nl_say "  $name: partial, resumable ($(grep -o '"progress": {[^}]*}' "$d/progress.json" 2>/dev/null | head -1))"; nl_mark "$name" failed "partial (resumable)"; return 3; fi
  if nl_judge_artifact "$name" "$rc" "$(art_of "$name")" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then
    nl_say "  $(grep -a "\"label\": \"hset-$name\"" "$RUNLOG" | tail -1 | cut -c1-260)"
    python3 $T/code/hermes_settings_paired.py "$basedir" "$d" --json "$d/paired-vs-baseline.json" 2>&1 | sed "s/^/  paired vs $(basename "$basedir"): /" | cut -c1-420
    return 0
  else nl_say "  $name: FAILED rc=$rc (124 at the hard stop = resumable)"; return 3; fi
}

for L in "${LEGS[@]}"; do IFS='|' read -r name est kind kv ctx think sets basedir <<< "$L"
  if done_leg "$name"; then nl_mark "$name" skipped "artifact present"; continue; fi
  if [ "$kind" = fit ]; then nl_mark fit running; run_fit; rc=$?; [ "$rc" = 0 ] && nl_mark fit done "$(python3 -c 'import json;print(json.load(open("'$RES'/fit/summary.json"))["n"])') cells" || nl_mark fit failed "partial (resumable)"
  else run_drv "$name" "$kv" "$ctx" "$think" "$sets" "$basedir"; fi
done

# paired readout over everything that finished (CPU only, idempotent)
pair() { [ -f "$1/summary.json" ] && [ -f "$2/summary.json" ] && { echo "### $(basename "$1") vs $(basename "$2")"; python3 $T/code/hermes_settings_paired.py "$2" "$1"; }; }
{ echo "# hermes-settings paired readout, $(date -u +%FT%TZ); each line: base = second dir, other = first dir"
  pair $RES/s5-on-f16-r2 $BASE_S5_ON; pair $RES/s5-on-q8 $BASE_S5_ON; pair $RES/s5-on-q8 $RES/s5-on-f16-r2
  pair $RES/s5-off-f16 $BASE_S5_ON
  pair $RES/st-on-f16-r2 $BASE_ST_ON; pair $RES/st-on-q8 $BASE_ST_ON; pair $RES/st-on-q8 $RES/st-on-f16-r2
  pair $RES/s5-on-f16-c32k $BASE_S5_ON; pair $RES/s5-on-f16-c64k $BASE_S5_ON; pair $RES/s5-on-f16-c128k $BASE_S5_ON
  pair $RES/s5-on-q4 $BASE_S5_ON; pair $RES/s5-on-q4 $RES/s5-on-q8
} > "$RES/paired-readout.txt" 2>&1
nl_say "paired readout: $RES/paired-readout.txt ($(grep -c '^###' "$RES/paired-readout.txt") pairs)"

for L in "${LEGS[@]}"; do IFS='|' read -r name _ <<< "$L"; done_leg "$name" || { nl_say "legs remain, resumable"; exit 3; }; done
nl_say "all legs done"; exit 0
