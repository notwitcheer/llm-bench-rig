#!/bin/bash
# 65s-sub10-core.sh: idle-queue item (the core-four half of 67, split 2026-10-11 ~14:45 CEST so the trending Saluki claims check
# lands this week: "65s" sorts before 66-bonsai2-phase2, so the next bench block runs this first and Bonsai GPQA resumes after).
# ROW-MAJOR (unlike 67): Saluki gets all four tasks first, the trending row the post needs; GSQ-RCO after.
# original 67 header follows.
# 67-sub10-qwen38-27b.sh: idle-queue item (2026-10-11 ~14:30 CEST, mercury, operator in the GPU thread: "make sure we
# continue to make the gpu work correctly during heures creuses").
# the sub-10 GB Qwen3.8-27B study: two more small-file rows next to Ternary-Bonsai-2-27B (65/66), on the exact board cells
# the Qwen3.8-27B Q6_K / UD-IQ3_XXS rows already have:
#   ConwayResearch/Underdog-Saluki-27B-1.0 IQ2-mix (7.89 GB, created 2026-10-08, apache-2.0, stock llama.cpp; its card
#     claims IFEval 93.5, MBPP+ 78.0, AIME 2026 avg@4 80.0, and compares itself with Bonsai 2)
#   ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF IQ2_XS (8.42 GB, the GSQ-RCO base Saluki builds on; asked for by a reader under
#     the 2026-09-12 card-fit post: "try GSQ-RCO, better than unsloth quants on agentic work")
# cells, task-major, the four second-tier tasks first so both rows have the core board before the long legs:
#   math500, humaneval_plus, mbpp_plus, ifeval, gpqa: think-on, max_tokens 16384, ctx 24576 -> results/<slug>-thinkon/
#   ma_aime_2026, ma_hmmt_feb_2026: think-on, max_tokens 32768, ctx 40960, exact_veto=True -> results/<slug>-matharena-32k/
# server: the rig's mainline llama-server (config.yaml llama_cpp.server_bin), no override. bodies = 65 (second-tier calls)
# + 60 (GPQAEval) + 59 (MathArenaEval, veto, dataset probes). resumable per item, exact-artifact skips, exit 3 while legs remain.
set -uo pipefail
NL_TAG=sub10core
MANIFEST=/home/witcheer/sub10-core-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
RUNLOG=/home/witcheer/sub10-run.log
TASKS="math500 humaneval_plus mbpp_plus ifeval"
ROWS="
underdog-saluki-27b-iq2-mix:/home/witcheer/models/saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf
qwen3-8-27b-gsq-rco-iq2-xs:/home/witcheer/models/gsq-rco-27b/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf
"
art_of() { # <slug> <task>
  case "$2" in
    gpqa) echo "$RIG/results/$1-thinkon/gpqa.json" ;;
    ma_*) echo "$RIG/results/$1-matharena-32k/${2}_detail.json" ;;
    *)    echo "$RIG/results/$1-thinkon/${2}_detail.json" ;;
  esac
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py" "$RIG/lib/evals/gpqa.py" "$RIG/lib/evals/matharena.py"
nl_check_disk 30
(cd $RIG && $PY -c "from lib.qwen_math_grader import grade; assert grade(r'\\frac{1}{2}','0.5')") && nl_say "  ok: math grader imports" || { nl_say "  math grader broken"; exit 2; }
$PY -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); assert len(d)==198" >/dev/null 2>&1 && nl_say "  ok: gpqa_diamond loads (198)" || { nl_say "  gpqa dataset probe FAILED"; exit 2; }
$PY -c "import inspect,sys; sys.path.insert(0,'$RIG'); from lib.evals.gpqa import GPQAEval; assert 'max_tokens' in inspect.signature(GPQAEval).parameters" && nl_say "  ok: GPQAEval(max_tokens=)" || { nl_say "  GPQAEval lacks max_tokens"; exit 2; }
$PY -c "import sys; sys.path.insert(0,'$RIG'); from latex2sympy2_extended import latex2sympy; from lib.evals.matharena import grade_with_timeout as g; assert g('50', r'50(1 - \\frac{1}{2^{101} - 1})', veto=True, return_vetoed=True)[2]; assert g('277','277', veto=True)[0]" >/dev/null 2>&1 && nl_say "  ok: exact-number veto" || { nl_say "  exact-number veto check FAILED"; exit 2; }
$PY -c "
from datasets import load_dataset
want={'aime_2026':30,'hmmt_feb_2026':33}
for k,n in want.items(): assert len(load_dataset('MathArena/'+k,split='train'))==n,k
" >/dev/null 2>&1 && nl_say "  ok: two MathArena contest sets (63 items)" || { nl_say "  matharena dataset probe FAILED"; exit 2; }
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
for row in $ROWS; do for t in $TASKS; do slug="${row%%:*}"; path="${row#*:}"
  art=$(art_of "$slug" "$t")
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 600 ] || { nl_say "window exhausted before $slug $t ($left s left), resumable"; nl_mark "${slug}_$t" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_mark "${slug}_$t" running; nl_say "=== $slug $t THINK-ON (budget $((left/60)) min) ==="
  timeout "$left" $PY -u - "$path" "$slug" "$t" >> "$RUNLOG" 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
from lib.evals.base import LLMClient
path, slug, t = sys.argv[1], sys.argv[2], sys.argv[3]
if t.startswith("ma_"):
    think_max, ctx, rd, out = 32768, 40960, Path("results") / f"{slug}-matharena-32k", f"{t}_detail.json"
elif t == "gpqa":
    think_max, ctx, rd, out = 16384, 24576, Path("results") / f"{slug}-thinkon", "gpqa.json"
else:
    think_max, ctx, rd, out = 16384, 24576, Path("results") / f"{slug}-thinkon", f"{t}_detail.json"
rd.mkdir(parents=True, exist_ok=True)
print(f"[{slug}] server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=ctx))}", flush=True)
proc = q.start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=3600) as client:
        t0 = time.time()
        if t == "ifeval":
            from lib.evals.ifeval import IFEvalEval
            r = IFEvalEval(client=client, results_dir=rd, max_tokens=think_max).evaluate()
        elif t == "math500":
            from lib.evals.math500 import Math500Eval
            r = Math500Eval(client=client, results_dir=rd, max_tokens=think_max).evaluate()
        elif t in ("humaneval_plus", "mbpp_plus"):
            from lib.evals.evalplus import EvalPlusEval
            r = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=think_max).evaluate()
        elif t == "gpqa":
            from lib.evals.gpqa import GPQAEval
            r = GPQAEval(client=client, results_dir=rd, max_tokens=think_max).evaluate()
        else:
            from lib.evals.matharena import MathArenaEval
            r = MathArenaEval(client=client, task=t, results_dir=rd, max_tokens=think_max, exact_veto=True).evaluate()
        r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
        r["bench"] = "sub-10 GB Qwen3.8-27B study (2026-10-11)"
        (rd / out).write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 2
  if nl_judge_artifact "${slug}_$t" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a "\[$slug\] $t:" "$RUNLOG" | tail -1 | cut -c1-220)"; else nl_say "  $slug $t: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done; done
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; [ -f "$(art_of "$slug" "$t")" ] || { nl_say "legs remain, resumable"; exit 3; }; done; done
nl_say "all legs done"; exit 0
