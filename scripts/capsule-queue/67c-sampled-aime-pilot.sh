#!/bin/bash
# 67c-sampled-aime-pilot.sh: idle-queue item (2026-10-20 ~17:45 CEST, mercury, GPU thread, operator "yes").
# greedy-vs-sampled pilot: AIME 2026 (30), K=8 sampled completions per problem (T 0.6, top_p 0.95, top_k 20, the
# Qwen3.8 card's thinking-mode settings), think-on, max_tokens 32768, 40,960 ctx per slot, same grader + exact veto
# as the greedy rows in results/<slug>-matharena-32k/. why: DeepSeek's DeepInfer eval reports AIME 2026 as the mean of
# 32 samples, GPQA as the mean of 8 (github.com/deepseek-ai/DeepInfer eval/README.md, read 2026-10-20); our boards
# score one greedy pass. driver: ~/benchmark-rig/scripts/sampled_contest.py (vault copy scripts/), smoke-tested
# 2026-10-20 17:36 on PQ2_0 (2 problems x 2 samples, np 2, port 8199).
# concurrency (probed 2026-10-20 17:2x, 1500-token sampled requests): Q6_K -np 3 = 162 tok/s aggregate (1 stream 62),
# 29.4 GB; -np 4 loads at 32.1 GB and OOMs on the first concurrent batch, so 3. PQ2_0 (prism) -np 8 = 457 tok/s, 29.1 GB.
# sizing (estimate from the greedy legs' token totals x8): PQ2_0 ~3.8M tokens / ~450 = ~2.5-3 h; Q6_K ~4.0M / ~160 = ~7 h.
# resumable per sample (<task>_samples.json); exact-artifact skips; exit 3 while legs remain.
set -uo pipefail
NL_TAG=sampaime
MANIFEST=/home/witcheer/sampled-aime-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
RUNLOG=/home/witcheer/sampled-aime-run.log
DRIVER=$RIG/scripts/sampled_contest.py
K=8
ROWS="
ternary-bonsai-2-27b-pq2-0:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf:8:prism
qwen3-8-27b-q6-k:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf:3:mainline
"
art_of() { echo "$RIG/results/$1-matharena-32k-s$K/ma_aime_2026_sampled_detail.json"; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$PRISM" "$DRIVER" "$RIG/lib/evals/matharena.py"
$PRISM --version 2>&1 | grep -q "4fbda12" && nl_say "  ok: prism build 4fbda12" || { nl_say "  prism build changed or missing"; exit 2; }
nl_check_disk 30
$PY -c "from datasets import load_dataset; assert len(load_dataset('MathArena/aime_2026',split='train'))==30" >/dev/null 2>&1 && nl_say "  ok: aime_2026 loads (30)" || { nl_say "  aime dataset probe FAILED"; exit 2; }
(cd $RIG && PYTHONPATH=. $PY -c "import scripts.sampled_contest" 2>/dev/null || PYTHONPATH=. $PY -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('sc','$DRIVER'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)") && nl_say "  ok: driver imports" || { nl_say "  driver import FAILED"; exit 2; }
PH=""; RUNS=0
for row in $ROWS; do IFS=: read -r slug path np bin <<< "$row"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; exit 2; }
  PH="$PH ${slug}_s$K"; [ -f "$(art_of "$slug")" ] && nl_say "  $slug aime s$K: present" || { nl_say "  $slug aime s$K: RUN (np $np, $bin)"; RUNS=$((RUNS+1)); }; done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS (hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"
for row in $ROWS; do IFS=: read -r slug path np bin <<< "$row"
  art=$(art_of "$slug")
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $slug ($left s left), resumable"; nl_mark "${slug}_s$K" failed "window exhausted (resumable)"; continue; }
  EXTRA=""; [ "$bin" = prism ] && EXTRA="--server-bin $PRISM --server-label PrismML-Eng/llama.cpp_prism_4fbda12"
  t0=$(date +%s); nl_mark "${slug}_s$K" running; nl_say "=== $slug aime K=$K np=$np (budget $((left/60)) min) ==="
  timeout "$left" $PY -u "$DRIVER" --model "$path" --slug "$slug" --task ma_aime_2026 --k $K --np "$np" --port 8090 $EXTRA >> "$RUNLOG" 2>&1
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if nl_judge_artifact "${slug}_s$K" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "DONE $slug" "$RUNLOG" | tail -1 | cut -c1-240)"; else nl_say "  $slug: not complete rc=$rc (124 at the hard stop = resumable), $(grep -ac "^\[sampled\] [0-9]" "$RUNLOG") sample lines so far"; fi
done
for row in $ROWS; do slug="${row%%:*}"; [ -f "$(art_of "$slug")" ] || { nl_say "legs remain, resumable"; exit 3; }; done
nl_say "all legs done"; exit 0
