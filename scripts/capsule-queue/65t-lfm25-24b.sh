#!/bin/bash
# 65t-lfm25-24b.sh: idle-queue item, STAGED 2026-10-14 by mercury (subagent). sorts after 65s-sub10-core, before 66-bonsai2-phase2.
# LiquidAI LFM2.5-24B-A2B (created 2026-10-12, lfm2moe, 24.0B total / 2.45B active, 64 experts top-4, always thinks)
# on the second-tier THINK-ON board, same regime and same items as the comparator Liquid names on its card,
# Qwen3.6-35B-A3B (board row qwen3-6-35b-a3b-ud-q5-k-m-thinkon): ifeval, gpqa, math500, humaneval_plus, mbpp_plus,
# think-on, max_tokens 16384, ctx 24576, zero-shot, greedy, same eval classes, same harness build (~/llama.cpp-nl).
# comparator 16k cells: ifeval 87.25 (cap 4.3%), gpqa 76.77 (cap 9.1%), math500 72.8 (cap 30.2%, budget-limited),
# humaneval+ 95.12 (cap 0%), mbpp+ 81.22 (cap 1.1%). Liquid's card (vLLM, temp 0.6, 32k): IFEval 96.75 vs 93.06,
# GPQA 74.42 vs 83.69, MATH-500 96.04 vs 96.73.
# rows: unsloth UD-Q5_K_XL (17.9 GB, headline, same bit tier as the comparator's UD-Q5_K_M) then LiquidAI Q8_0 (25.5 GB,
# vendor near-lossless point). no fork: the harness build 5d16e81dd loads lfm2moe and tokenises the 128K vocab
# identically to the HF tokenizer (5/5 strings, 2026-10-14 CPU check). sizing from the 2026-10-14 probe: see the vault
# note 99-personal/benchmark-rig/lfm25-24b-status.md.
# legs TASK-MAJOR (for each task, both rows), headline set first. resumable per item (<task>_progress.json /
# gpqa_progress.json), exact-artifact skips, hard stop = end of the heures creuses block, exit 3 while legs remain.
set -uo pipefail
NL_TAG=lfm25
MANIFEST=/home/witcheer/lfm25-24b-status.json
RUNLOG=/home/witcheer/lfm25-24b-run.log
THINK_MAX=16384
CTX=24576
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
SRV=/home/witcheer/llama.cpp-nl/build/bin/llama-server
SRV_COMMIT=5d16e81dd
TASKS="ifeval gpqa math500 humaneval_plus mbpp_plus"
ROWS="
lfm2-5-24b-a2b-ud-q5-k-xl:/home/witcheer/models/lfm25-24b/LFM2.5-24B-A2B-UD-Q5_K_XL.gguf
lfm2-5-24b-a2b-q8-0:/home/witcheer/models/lfm25-24b-liquid/LFM2.5-24B-A2B-Q8_0.gguf
"
art_of() { # <slug> <task>
  if [ "$2" = gpqa ]; then echo "$RIG/results/$1-thinkon/gpqa.json"; else echo "$RIG/results/$1-thinkon/${2}_detail.json"; fi
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$SRV" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py" "$RIG/lib/evals/gpqa.py"
$SRV --version 2>&1 | grep -q "$SRV_COMMIT" && nl_say "  ok: harness build $SRV_COMMIT" || { nl_say "  harness llama-server build changed or missing (want $SRV_COMMIT)"; exit 2; }
(cd $RIG && $PY -c "from lib.qwen_math_grader import grade; assert grade(r'\\frac{1}{2}','0.5')") && nl_say "  ok: math grader imports" || { nl_say "  math grader broken"; exit 2; }
$PY -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); assert len(d)==198" >/dev/null 2>&1 && nl_say "  ok: gpqa_diamond loads (198)" || { nl_say "  gpqa dataset probe FAILED"; exit 2; }
$PY -c "
from datasets import load_dataset
for name, cfg, split, n in [('google/IFEval', None, 'train', 541), ('HuggingFaceH4/MATH-500', None, 'test', 500), ('evalplus/humanevalplus', None, 'test', 164), ('evalplus/mbppplus', None, 'test', 378)]:
    assert len(load_dataset(name, cfg, split=split)) == n, name
" >/dev/null 2>&1 && nl_say "  ok: ifeval / math500 / humaneval+ / mbpp+ datasets load" || { nl_say "  second-tier dataset probe FAILED"; exit 2; }
$PY -c "import inspect,sys; sys.path.insert(0,'$RIG'); from lib.evals.gpqa import GPQAEval; assert 'max_tokens' in inspect.signature(GPQAEval).parameters" && nl_say "  ok: GPQAEval(max_tokens=)" || { nl_say "  GPQAEval lacks max_tokens"; exit 2; }
nl_check_disk 30
PH=""; RUNS=0
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE $path"; exit 2; }
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
  timeout "$left" $PY -u - "$path" "$slug" "$t" "$THINK_MAX" "$CTX" >> "$RUNLOG" 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
from lib.evals.base import LLMClient
path, slug, t, think_max, ctx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rd = Path("results") / f"{slug}-thinkon"; rd.mkdir(parents=True, exist_ok=True)
out = "gpqa.json" if t == "gpqa" else f"{t}_detail.json"
print(f"[{slug}] server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=ctx))}", flush=True)
proc = q.start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=1800) as client:
        t0 = time.time()
        if t == "gpqa":
            from lib.evals.gpqa import GPQAEval
            ev = GPQAEval(client=client, results_dir=rd, max_tokens=think_max)
        elif t == "ifeval":
            from lib.evals.ifeval import IFEvalEval
            ev = IFEvalEval(client=client, results_dir=rd, max_tokens=think_max)
        elif t == "math500":
            from lib.evals.math500 import Math500Eval
            ev = Math500Eval(client=client, results_dir=rd, max_tokens=think_max)
        else:
            from lib.evals.evalplus import EvalPlusEval
            ev = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=think_max)
        r = ev.evaluate()
        r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
        r["bench"] = "lfm25-24b vs Qwen3.6-35B-A3B second-tier think-on 16k (same items, same harness build)"
        r["server_bin"] = "ggml-org/llama.cpp b10371 (5d16e81dd), ~/llama.cpp-nl harness build"
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
