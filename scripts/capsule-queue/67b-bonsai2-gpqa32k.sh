#!/bin/bash
# 67b-bonsai2-gpqa32k.sh: idle-queue item (2026-10-20 ~07:50 CEST, mercury, GPU thread, operator "ok go").
# GPQA-diamond think-on at max_tokens 32768, ctx 40960 for both Ternary-Bonsai-2-27B rows, whose 16k legs capped
# 32-34% (PTQ1_0 64/198, PQ2_0 67/198): a floor, not a score. same treatment as 60-gpqa-thinkon-32k.sh gave the
# three rows that capped >10% at 16k. -> results/<slug>-thinkon-32k/gpqa.json, never the 16k dir.
# body: 66-bonsai2-phase2.sh (prism server_bin override, same harness). ONLY change: task gpqa only, budget/ctx/dir.
# sizing (estimate): ~160 tok/s; capped items up to 32k each -> ~4-5 h per row, ~9-10 h both. sorts after 67-sub10.
# resumable per item (gpqa_progress.json), exact-artifact skips, exit 3 while legs remain.
set -uo pipefail
NL_TAG=bonsai2g32
MANIFEST=/home/witcheer/bonsai2-gpqa32k-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PRISM=/home/witcheer/llama.cpp-prism/build/bin/llama-server
RUNLOG=/home/witcheer/bonsai2-gpqa32k-run.log
TASKS="gpqa"
ROWS="
ternary-bonsai-2-27b-ptq1-0:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf
ternary-bonsai-2-27b-pq2-0:/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf
"
art_of() { # <slug> <task>
  echo "$RIG/results/$1-thinkon-32k/gpqa.json"
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$PRISM" "$RIG/lib/quality.py" "$RIG/lib/evals/gpqa.py"
$PRISM --version 2>&1 | grep -q "4fbda12" && nl_say "  ok: prism build 4fbda12" || { nl_say "  prism build changed or missing"; exit 2; }
nl_check_disk 30
$PY -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); assert len(d)==198" >/dev/null 2>&1 && nl_say "  ok: gpqa_diamond loads (198)" || { nl_say "  gpqa dataset probe FAILED"; exit 2; }
$PY -c "import inspect,sys; sys.path.insert(0,'$RIG'); from lib.evals.gpqa import GPQAEval; assert 'max_tokens' in inspect.signature(GPQAEval).parameters" && nl_say "  ok: GPQAEval(max_tokens=)" || { nl_say "  GPQAEval lacks max_tokens"; exit 2; }
PH=""; RUNS=0
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; exit 2; }
  PH="$PH ${slug}_$t"; [ -f "$(art_of "$slug" "$t")" ] && nl_say "  $slug $t: present" || { nl_say "  $slug $t: RUN"; RUNS=$((RUNS+1)); }; done; done
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
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"
  art=$(art_of "$slug" "$t")
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $slug $t ($left s left), resumable"; nl_mark "${slug}_$t" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_mark "${slug}_$t" running; nl_say "=== $slug $t THINK-ON (budget $((left/60)) min) ==="
  timeout "$left" $PY -u - "$path" "$slug" "$t" "$PRISM" >> "$RUNLOG" 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
path, slug, t, prism = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
_orig = q.get
q.get = lambda k, *a, **kw: prism if k == "llama_cpp.server_bin" else _orig(k, *a, **kw)
from lib.evals.base import LLMClient
think_max, ctx, rd, out = 32768, 40960, Path("results") / f"{slug}-thinkon-32k", "gpqa.json"
rd.mkdir(parents=True, exist_ok=True)
print(f"[{slug}] server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=ctx))}", flush=True)
proc = q.start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=10800) as client:
        t0 = time.time()
        from lib.evals.gpqa import GPQAEval
        r = GPQAEval(client=client, results_dir=rd, max_tokens=think_max).evaluate()
        r["bench"] = "gpqa-thinkon-32k (bonsai2 rows added 2026-10-20; 16k legs capped >10%)"
        r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
        r["server_bin"] = "PrismML-Eng/llama.cpp prism 4fbda12 (CUDA 12.8, sm_120)"
        (rd / out).write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if nl_judge_artifact "${slug}_$t" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "\[$slug\] $t:" "$RUNLOG" | tail -1 | cut -c1-220)"; else nl_say "  $slug $t: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done; done
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; [ -f "$(art_of "$slug" "$t")" ] || { nl_say "legs remain, resumable"; exit 3; }; done; done
nl_say "all legs done"; exit 0
