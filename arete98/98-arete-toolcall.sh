#!/bin/bash
# 98-arete-toolcall.sh: idle-queue item (2026-11-07).
# WHAT: toolcall-v1 (BFCL v4 raw-mode slice, frozen 2026-10-15) on NousResearch Arete-27B Q6_K (bartowski), legs st-on (S1-S4, 899 requests)
#   and s5-on (100 multi-turn entries). Same harness, build (mainline b10371 5d16e81dd), parser-2, greedy, ctx, max_tokens and frozen ids as
#   the base row qwen3-8-27b-q6-k (Unsloth Q6_K, run 2026-10-15/16: st-on 799 = 88.2, s5-on 81/100). Copy of 67d with ROWS/LEGS/paths changed only.
# WHY: Nous says the Arete values post-train cost no capability (Hermes Bench 65.77 -> 65.60). item 96's board found no loss on GPQA/HE/ARC/MMLU;
#   this asks the Hermes-shaped question: does the fine-tune still drive multi-turn tool calls like its base, on the file people download.
# template check (2026-11-07, CPU): the Arete GGUF embeds the OFFICIAL Qwen3.8 chat template byte for byte (md5 519239a4 = qwen38-tokenizer/
#   chat_template.jinja); tokenizer tokens/merges/types/pre/eos identical to the base Q6_K (padding id differs, unused in raw /completion).
#   raw mode renders client-side with the official template anyway; --tokenize-all proves /tokenize parity per leg.
# PRE-REGISTERED VERDICT (paired exact McNemar on the same frozen ids, Arete vs base Q6_K, written before any Arete output):
#   primary = s5-on. p < 0.05 and Arete higher -> ARETE AHEAD; p < 0.05 and Arete lower -> ARETE BELOW; otherwise LEVEL (no measurable difference on one greedy run). st-on reported the same way, secondary.
#   confounds stated in any copy: one greedy run each; different GGUF makers (Unsloth vs bartowski); base row ran 3 weeks earlier on the same build.
# window: weekend HC; own hard stop Sat 23:40 local so the 09z30 duck item (NOT_BEFORE Sun 00:00) starts on time; resumable (exit 3) per leg.
# card estimate (probe pace of the base Q6_K row): st-on ~81 min + s5-on ~103 min = ~3.1 h + loads; ESTIMATE.
set -uo pipefail
NL_TAG=arete98
MANIFEST=/home/witcheer/arete98-toolcall-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
T=/home/witcheer/toolcall-v1
PY=$T/.venv/bin/python
RES=/home/witcheer/benchmark-rig/results/toolcall-v1
RUNLOG=/home/witcheer/arete98-toolcall-run.log
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
PORT=8097
LEGS="st-on s5-on"
ROWS="
arete-27b-q6-k:main:/home/witcheer/models/arete-27b/NousResearch_Arete-27B-Q6_K.gguf
"
art_of() { echo "$RES/$1-$2/summary.json"; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$MAIN" "$PRISM" "$T/code/driver.py" "$T/frozen/toolcall-v1/manifest.json" "$T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz" "$T/qwen38-tokenizer/tokenizer.json"
nl_check_disk 20
for L in st-on s5-on; do [ -f "$RES/qwen3-8-27b-q6-k-$L/summary.json" ] && nl_say "  ok: base row qwen3-8-27b-q6-k $L present" || { nl_say "  base row $L missing"; exit 2; }; done
T8=$(md5sum < /home/witcheer/toolcall-v1/artifacts/arete/arete_embedded_template.jinja | cut -c1-8); [ "$T8" = 519239a4 ] && nl_say "  ok: Arete embedded template = official (519239a4)" || { nl_say "  Arete template dump missing/changed ($T8)"; exit 2; }
( cd $T/frozen/toolcall-v1 && sha256sum -c --quiet FROZEN.sha256 ) && nl_say "  ok: frozen list sha256 intact" || { nl_say "  frozen list changed"; exit 2; }
( cd $T/code && $PY -c "import qwen38_xml as X; assert X.PARSER_VERSION == 'toolcall-v1-parser-2', X.PARSER_VERSION" 2>/dev/null ) && nl_say "  ok: parser toolcall-v1-parser-2" || { nl_say "  parser version changed (re-run roundtrip_test.py and update this item)"; exit 2; }
case "$($PRISM --version 2>&1)" in *4fbda12*) nl_say "  ok: prism 4fbda12";; *) nl_say "  prism build changed"; exit 2;; esac
case "$($MAIN --version 2>&1)" in *5d16e81*) nl_say "  ok: mainline 5d16e81";; *) nl_say "  mainline build changed"; exit 2;; esac
PH=""; RUNS=0
for leg in $LEGS; do for row in $ROWS; do slug=${row%%:*}; rest=${row#*:}; path=${rest#*:}
  [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; exit 2; }
  PH="$PH ${slug}_$leg"; [ -f "$(art_of "$slug" "$leg")" ] && nl_say "  $slug $leg: present" || { nl_say "  $slug $leg: RUN"; RUNS=$((RUNS+1)); }; done; done
CAP=$(date -d "2026-11-07 23:40" +%s); [ -z "${NL_HARD_STOP_EPOCH:-}" ] && export NL_HARD_STOP_EPOCH=$CAP; [ "$NL_HARD_STOP_EPOCH" -gt "$CAP" ] && export NL_HARD_STOP_EPOCH=$CAP
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
  for i in $(seq 1 150); do H=$(curl -s -m 3 http://127.0.0.1:$PORT/health || true); [[ ${H,,} == *ok* ]] && break; kill -0 $sp 2>/dev/null || break; sleep 2; done
  H=$(curl -s -m 3 http://127.0.0.1:$PORT/health || true); if [[ ${H,,} != *ok* ]]; then nl_say "  $slug: server did not come up"; kill $sp 2>/dev/null; nl_mark "${slug}_$leg" failed "server load"; continue; fi
  ( cd $T/code && timeout $(( left + 60 )) $PY -u driver.py --server http://127.0.0.1:$PORT --label "$slug-$leg" --sets $sets \
      --manifest $T/frozen/toolcall-v1/manifest.json --tokenizer $T/qwen38-tokenizer --prompts $T/artifacts/prompts/prompts_toolcall-v1.jsonl.gz \
      --out "$d" --think $think --max-tokens 16384 --ctx $ctx --budget-s $left --tokenize-all --server-bin "$blabel" >> "$RUNLOG" 2>&1 )
  rc=$?
  kill $sp 2>/dev/null; sleep 3; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if [ "$rc" = 3 ]; then nl_say "  $slug $leg: partial, resumable ($(grep -o '"progress": {[^}]*}' "$d/progress.json" | head -1))"; nl_mark "${slug}_$leg" failed "partial (resumable)"; continue; fi
  if nl_judge_artifact "${slug}_$leg" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "\"label\": \"$slug-$leg\"" "$RUNLOG" | tail -1 | cut -c1-260)"; else nl_say "  $slug $leg: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done; done
for leg in $LEGS; do for row in $ROWS; do slug=${row%%:*}; [ -f "$(art_of "$slug" "$leg")" ] || { nl_say "legs remain, resumable"; exit 3; }; done; done
PAIR=$RES/arete98-paired.json
( cd $T/code && $PY paired_nvfp4.py "$PAIR" "arete=$RES/arete-27b-q6-k-{leg}" "base=$RES/qwen3-8-27b-q6-k-{leg}" ) >> "$RUNLOG" 2>&1 && nl_say "  paired: $(tr -d '\n' < "$PAIR" | grep -o '"pairs".*' | cut -c1-400)"
nl_say "all legs done"; exit 0
