#!/bin/bash
# 65-bonsai2-secondtier.sh: idle-queue item (2026-10-08 ~08:05 CEST, mercury, operator "go" in the GPU thread).
# Ternary-Bonsai-2-27B (prism-ml, base Qwen3.8-27B, ternary weights) on the second-tier THINK-ON board, same regime as the
# Qwen3.8-27B Q6_K / UD-IQ3_XXS rows it is compared with: ifeval, math500, humaneval_plus, mbpp_plus, think-on, max_tokens 16384,
# ctx 24576, zero-shot, greedy, same eval classes. ONLY change: llama_cpp.server_bin -> the PrismML fork (~/llama.cpp-prism,
# prism branch 4fbda12, CUDA 12.8, sm_120; PTQ1_0/PQ2_0 kernels are not in mainline). two rows: PTQ1_0 (5.95 GB) and PQ2_0 (7.21 GB).
# probe 2026-10-08 07:58 CEST (~/bonsai-probe.sh): ~160 tok/s, 8.3 GB VRAM, math500 5/5, ifeval 3/3, humaneval+ 3/3, mbpp+ 3/3,
# 0 capped, token counts on the same items within the Q6_K row's range (thinking active). estimate ~4-6 h per row for all four tasks.
# legs TASK-MAJOR (for each task, both rows) so a cut study has every task for the headline row first. resumable per item
# (<task>_progress.json), exact-artifact skips, hard stop = end of the heures creuses block, exit 3 while legs remain.
set -uo pipefail
NL_TAG=bonsai2
MANIFEST=/home/witcheer/bonsai2-secondtier-status.json
THINK_MAX=16384
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
TASKS="math500 humaneval_plus mbpp_plus ifeval"
ROWS="
ternary-bonsai-2-27b-ptq1-0:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf
ternary-bonsai-2-27b-pq2-0:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf
"
nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$PRISM" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py"
$PRISM --version 2>&1 | grep -q "4fbda12" && nl_say "  ok: prism build 4fbda12" || { nl_say "  prism build changed or missing"; exit 2; }
(cd $RIG && $PY -c "from lib.qwen_math_grader import grade; assert grade(r'\\frac{1}{2}','0.5')") && nl_say "  ok: math grader imports" || { nl_say "  math grader broken"; exit 2; }
nl_check_disk 30
PH=""; RUNS=0
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; exit 2; }
  PH="$PH ${slug}_$t"; [ -f "$RIG/results/$slug-thinkon/${t}_detail.json" ] && nl_say "  $slug $t: present" || { nl_say "  $slug $t: RUN"; RUNS=$((RUNS+1)); }; done; done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS (hard stop $(date -u -d "@$DL" "+%H:%M") UTC, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(nl_deadline_epoch)
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"
  art="$RIG/results/$slug-thinkon/${t}_detail.json"
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $slug $t ($left s left), resumable"; nl_mark "${slug}_$t" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_mark "${slug}_$t" running; nl_say "=== $slug $t THINK-ON (budget $((left/60)) min) ==="
  timeout "$left" $PY -u - "$path" "$slug" "$THINK_MAX" "$t" "$PRISM" >> ~/bonsai2-run.log 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
path, slug, think_max, t, prism = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
_orig = q.get
q.get = lambda k, *a, **kw: prism if k == "llama_cpp.server_bin" else _orig(k, *a, **kw)
from lib.evals.base import LLMClient
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
rd = Path("results") / f"{slug}-thinkon"; rd.mkdir(parents=True, exist_ok=True)
print(f"[{slug}] server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=24576))}", flush=True)
proc = q.start_llama_server(path, ctx_size=24576)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=1800) as client:
        t0 = time.time()
        if t == "ifeval": ev = IFEvalEval(client=client, results_dir=rd, max_tokens=think_max)
        elif t == "math500": ev = Math500Eval(client=client, results_dir=rd, max_tokens=think_max)
        else: ev = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=think_max)
        r = ev.evaluate()
        r["regime"] = f"think-on, max_tokens {think_max}, ctx 24576, zero-shot, greedy"
        r["server_bin"] = "PrismML-Eng/llama.cpp prism 4fbda12 (CUDA 12.8, sm_120)"
        (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if nl_judge_artifact "${slug}_$t" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "\[$slug\] $t:" ~/bonsai2-run.log | tail -1 | cut -c1-220)"; else nl_say "  $slug $t: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done; done
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; [ -f "$RIG/results/$slug-thinkon/${t}_detail.json" ] || { nl_say "legs remain, resumable"; exit 3; }; done; done
nl_say "all legs done"; exit 0
