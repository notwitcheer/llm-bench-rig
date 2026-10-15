#!/bin/bash
# 68-toolcall-v1.sh: idle-queue item (2026-10-15, mercury; operator go "queue it" in the GPU thread after the step-1 probe).
# toolcall-v1 = BFCL v4 slice in raw mode (design: vault 99-personal/benchmark-rig/drafts/2026-10-15-toolcall-slice-design.md,
# status: toolcall-v1-status.md). Client-side official Qwen3.8 template -> llama-server /completion, greedy, parser
# toolcall-v1-parser-2, BFCL 2026.3.23 checkers unchanged. Six Qwen3.8-27B-family rows, legs task-major:
#   st-on  : S1..S4 single-turn (899 requests, S1+S2 first), think-on, max_tokens 16384, ctx 24576
#   st-off : same list, think-off (operator asked for it by name, Saluki's card claim is think-off)
#   s5-on  : S5 multi_turn_base phase 1 (100 entries), think-on, max_tokens 16384, ctx 40960
# results: ~/benchmark-rig/results/toolcall-v1/<slug>-<leg>/ (summary.json only when the leg is complete; progress.json
# otherwise). resumable per item inside a leg (items.jsonl / multi_turn.jsonl are appended), exit 3 while legs remain.
set -uo pipefail
NL_TAG=toolcall
MANIFEST=/home/witcheer/toolcall-v1-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
T=/home/witcheer/toolcall-v1
PY=$T/.venv/bin/python
RES=/home/witcheer/benchmark-rig/results/toolcall-v1
RUNLOG=/home/witcheer/toolcall-v1-run.log
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
PORT=8097
LEGS="st-on st-off s5-on"
ROWS="
qwen3-8-27b-q6-k:main:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
qwen3-8-27b-ud-iq3-xxs:main:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-UD-IQ3_XXS.gguf
underdog-saluki-27b-iq2-mix:main:/home/witcheer/models/saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf
qwen3-8-27b-gsq-rco-iq2-xs:main:/home/witcheer/models/gsq-rco-27b/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf
ternary-bonsai-2-27b-pq2-0:prism:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf
ternary-bonsai-2-27b-ptq1-0:prism:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf
"
art_of() { echo "$RES/$1-$2/summary.json"; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$MAIN" "$PRISM" "$T/code/driver.py" "$T/frozen/toolcall-v1/manifest.json" "$T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz" "$T/qwen38-tokenizer/tokenizer.json"
nl_check_disk 20
( cd $T/frozen/toolcall-v1 && sha256sum -c --quiet FROZEN.sha256 ) && nl_say "  ok: frozen list sha256 intact" || { nl_say "  frozen list changed"; exit 2; }
( cd $T/code && $PY -c "import qwen38_xml as X; assert X.PARSER_VERSION == 'toolcall-v1-parser-2', X.PARSER_VERSION" 2>/dev/null ) && nl_say "  ok: parser toolcall-v1-parser-2" || { nl_say "  parser version changed (re-run roundtrip_test.py and update this item)"; exit 2; }
case "$($PRISM --version 2>&1)" in *4fbda12*) nl_say "  ok: prism 4fbda12";; *) nl_say "  prism build changed"; exit 2;; esac
case "$($MAIN --version 2>&1)" in *5d16e81*) nl_say "  ok: mainline 5d16e81";; *) nl_say "  mainline build changed"; exit 2;; esac
PH=""; RUNS=0
for leg in $LEGS; do for row in $ROWS; do slug=${row%%:*}; rest=${row#*:}; path=${rest#*:}
  [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; exit 2; }
  PH="$PH ${slug}_$leg"; [ -f "$(art_of "$slug" "$leg")" ] && nl_say "  $slug $leg: present" || { nl_say "  $slug $leg: RUN"; RUNS=$((RUNS+1)); }; done; done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS (hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done

for leg in $LEGS; do for row in $ROWS; do slug=${row%%:*}; rest=${row#*:}; kind=${rest%%:*}; path=${rest#*:}
  art=$(art_of "$slug" "$leg")
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) - 120 ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $slug $leg ($left s left), resumable"; nl_mark "${slug}_$leg" failed "window exhausted (resumable)"; continue; }
  case $leg in st-on) think=on; sets=S1,S2,S3,S4; ctx=24576;; st-off) think=off; sets=S1,S2,S3,S4; ctx=24576;; s5-on) think=on; sets=S5; ctx=40960;; esac
  if [ "$kind" = prism ]; then bin=$PRISM; blabel="PrismML-Eng/llama.cpp prism 4fbda12"; else bin=$MAIN; blabel="ggml-org/llama.cpp mainline b10371 (5d16e81dd)"; fi
  d="$RES/$slug-$leg"; mkdir -p "$d"
  t0=$(date +%s); nl_mark "${slug}_$leg" running; nl_say "=== $slug $leg (ctx $ctx, budget $((left/60)) min) ==="
  "$bin" -m "$path" --port $PORT --n-gpu-layers 99 --host 127.0.0.1 --jinja --ctx-size $ctx >> "$d/server.log" 2>&1 &
  sp=$!
  for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && break; kill -0 $sp 2>/dev/null || break; sleep 2; done
  if ! curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok; then nl_say "  $slug: server did not come up"; kill $sp 2>/dev/null; nl_mark "${slug}_$leg" failed "server load"; continue; fi
  ( cd $T/code && timeout $(( left + 60 )) $PY -u driver.py --server http://127.0.0.1:$PORT --label "$slug-$leg" --sets $sets \
      --manifest $T/frozen/toolcall-v1/manifest.json --tokenizer $T/qwen38-tokenizer --prompts $T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz \
      --out "$d" --think $think --max-tokens 16384 --ctx $ctx --budget-s $left --tokenize-all --server-bin "$blabel" >> "$RUNLOG" 2>&1 )
  rc=$?
  kill $sp 2>/dev/null; sleep 3; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if [ "$rc" = 3 ]; then nl_say "  $slug $leg: partial, resumable ($(grep -o '"progress": {[^}]*}' "$d/progress.json" | head -1))"; nl_mark "${slug}_$leg" failed "partial (resumable)"; continue; fi
  if nl_judge_artifact "${slug}_$leg" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "\"label\": \"$slug-$leg\"" "$RUNLOG" | tail -1 | cut -c1-260)"; else nl_say "  $slug $leg: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done; done
for leg in $LEGS; do for row in $ROWS; do slug=${row%%:*}; [ -f "$(art_of "$slug" "$leg")" ] || { nl_say "legs remain, resumable"; exit 3; }; done; done
nl_say "all legs done"; exit 0
