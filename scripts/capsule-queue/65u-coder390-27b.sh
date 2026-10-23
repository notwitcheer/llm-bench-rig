#!/bin/bash
# 65u-coder390-27b.sh: idle-queue item, STAGED 2026-10-15 by mercury (subagent, operator-approved treatment that morning).
# sorts after 65t-lfm25-24b, before 66-bonsai2-phase2.
# matched-quant tune-vs-base treatment (Qwopus precedent): nerkyor Qwen3.8-27B-Coder390-EfficientThink (SFT + RLOO tune of
# Qwen3.8-27B, card: "remove unproductive reasoning tails") at Q6_K against the board's Qwen3.8-27B Q6_K think-on row
# (results/qwen3-8-27b-q6-k-thinkon/: math500 95.2 cap 1.6%, humaneval+ 92.07, mbpp+ 81.75 cap 3.7%, ifeval 89.65,
# gpqa 79.29). same tasks, same regime, same eval classes, same harness build, same server path as that row:
# think-on, max_tokens 16384, ctx 24576, zero-shot, greedy, lib.quality.start_llama_server defaults (-ngl 99 --jinja),
# NO spec decode: the base row ran without MTP/draft, so the embedded MTP head in this file stays unused.
# file: GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.gguf, 23,177,516,384 B, sha256 302597c5...76ab1 (HF API,
# repo sha 7235b052, read 2026-10-15). the embedded MTP head inflates the param count read from the file, so params are
# written explicitly per row (27.32 B, the base's own llama-bench figure) into every detail json.
# the efficiency claim maps to tokens_per_correct / completion tokens per item, tune vs base, from the <task>_tokens.jsonl
# sidecars (the base gpqa run predates the gpqa token sidecar; see the vault note coder390-27b-status.md).
# resumable per item (<task>_progress.json / gpqa_progress.json), exact-artifact skips, hard stop = end of the heures creuses
# block, exit 3 while legs remain. PLAN fails clean (exit 2) if the gguf is missing, still downloading, or the wrong size.
set -uo pipefail
NL_TAG=coder390
MANIFEST=/home/witcheer/coder390-27b-status.json
RUNLOG=/home/witcheer/coder390-27b-run.log
THINK_MAX=16384
CTX=24576
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
SRV=/home/witcheer/llama.cpp-nl/build/bin/llama-server
SRV_COMMIT=5d16e81dd
GGUF=/home/witcheer/models/coder390-27b/GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.gguf
GGUF_BYTES=23177516384
PARAMS="27.32 B"
TASKS="math500 humaneval_plus mbpp_plus ifeval gpqa"
ROWS="
qwen3-8-27b-coder390-q6-k-mtp:$GGUF
"
art_of() { # <slug> <task>
  if [ "$2" = gpqa ]; then echo "$RIG/results/$1-thinkon/gpqa.json"; else echo "$RIG/results/$1-thinkon/${2}_detail.json"; fi
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$SRV" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py" "$RIG/lib/evals/gpqa.py"
if [ ! -f "$GGUF" ]; then nl_say "  FAIL: gguf MISSING: $GGUF (download not finished or moved; see ~/c390-dl.log)"; exit 2; fi
got=$(stat -c %s "$GGUF")
if [ "$got" != "$GGUF_BYTES" ]; then nl_say "  FAIL: gguf INCOMPLETE or wrong file: $got B, want $GGUF_BYTES B ($GGUF)"; exit 2; fi
if ls /home/witcheer/models/coder390-27b/.cache/huggingface/download/GGUF/*.incomplete >/dev/null 2>&1; then nl_say "  FAIL: an hf download .incomplete is still present under coder390-27b/.cache, download not finished"; exit 2; fi
nl_say "  ok: gguf present, $got B (matches HF API size)"
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
  timeout "$left" $PY -u - "$path" "$slug" "$t" "$THINK_MAX" "$CTX" "$PARAMS" >> "$RUNLOG" 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
from lib.evals.base import LLMClient
path, slug, t, think_max, ctx, params = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
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
        r["bench"] = "coder390 tune vs Qwen3.8-27B Q6_K base, matched quant, second-tier think-on 16k (same items, same harness build, no spec decode)"
        r["server_bin"] = "ggml-org/llama.cpp b10371 (5d16e81dd), ~/llama.cpp-nl harness build"
        r["params"] = params
        r["params_note"] = "set explicitly: the Q6_K-MTP file embeds an MTP head that inflates the param count read from the gguf; 27.32 B is the base Qwen3.8-27B Q6_K llama-bench figure"
        r["model_file"] = "nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2 GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.gguf (sha256 302597c53d1b00f7230b53a8a050831390a4371e92a80c1ac644fa920af76ab1)"
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
