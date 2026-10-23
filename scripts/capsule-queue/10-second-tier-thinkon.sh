#!/bin/bash
# 10-second-tier-thinkon.sh: idle-queue item (heures creuses runner). resumable second-tier think-on legs, exact-slug skips.
# exits 3 while legs remain (stays in the queue), 0 once every leg has a fresh artifact (runner moves it to done/).
# (1) Gemma 4 31B Q4_0 64K long-context rerun with exception logging (artifact already present from 09-10, skips);
# (2) second-tier evals, THINK-ON (operator standing rule 2026-08-22; think-off is the extra leg, not the default), max_tokens 16384
# like the think-on GPQA legs, ctx 24576: ifeval, math500, humaneval_plus, mbpp_plus for the resident rows,
# artifacts results/<slug>-thinkon/<task>_detail.json (fresh dirs, no think-off artifact can be clobbered), exact-slug skips, hard stop = end of the heures creuses block, resumable.
# HC gate: the runner only fires inside heures creuses; hard stop = end of that block via nl_deadline_epoch.
set -uo pipefail
NL_TAG=secondtier
MANIFEST=/home/witcheer/secondtier-status.json
THINK_MAX=16384
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PORT=8090
TASKS="ifeval math500 humaneval_plus mbpp_plus"
ROWS="
qwen3-6-35b-a3b-ud-q5-k-m:/home/witcheer/models/qwen36-35b-base/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m:/home/witcheer/models/nemotron-35-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
ornith-1-5-35b-q4-k-m:/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf
ornith-9b:/home/witcheer/models/ornith-15-9b/Ornith-1.5-9B-Q6_K.gguf
qwen3-8-27b-q6-k:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
qwen3-6-27b-q6-k:/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf
qwopus3-8-27b-flash-q6-k:/home/witcheer/models/qwopus38-flash/Qwopus3.8-27B-Flash-MTP-Q6_K.gguf
qwable-27b-q4-k-m:/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf
gemma-4-31b-q4-0-it:/home/witcheer/models/gemma4-qat-q4/gemma-4-31B_q4_0-it.gguf
qwen3-8-27b-ud-iq3-xxs:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-UD-IQ3_XXS.gguf
"
GEMMA=/home/witcheer/models/gemma4-qat-q4/gemma-4-31B_q4_0-it.gguf
deadline_epoch() { nl_deadline_epoch; }   # HC-aware: runner exports NL_HARD_STOP_EPOCH = end of the current heures creuses block

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py" "$GEMMA"
nl_check_disk 50
grep -q "EXCEPTION" "$RIG/lib/agentic/longcontext.py" && nl_say "  ok: longcontext.py logs exceptions" || { nl_say "  longcontext.py lacks the exception log"; exit 2; }
$PY -c "import sys; sys.path.insert(0,'$RIG'); from lib.quality import SECOND_TIER_TASKS; assert {'ifeval','math500','humaneval_plus','mbpp_plus'} <= SECOND_TIER_TASKS" && nl_say "  ok: second-tier tasks registered" || { nl_say "  second-tier tasks missing"; exit 2; }
$PY -c "from datasets import load_dataset; assert len(load_dataset('google/IFEval',split='train'))==541; assert len(load_dataset('HuggingFaceH4/MATH-500',split='test'))==500; assert len(load_dataset('evalplus/humanevalplus',split='test'))==164; assert len(load_dataset('evalplus/mbppplus',split='test'))==378" >/dev/null 2>&1 && nl_say "  ok: four datasets" || { nl_say "  dataset probe FAILED"; exit 2; }
PH="gemma64k"; RUNS=0
[ -f "$RIG/results/gemma-4-31b-q4-0-it/longcontext-64k-rerun/longcontext_detail.json" ] && nl_say "  gemma64k: present" || { nl_say "  gemma64k: RUN"; RUNS=$((RUNS+1)); }
for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; continue; }
  for t in $TASKS; do PH="$PH ${slug}_$t"; [ -f "$RIG/results/$slug-thinkon/${t}_detail.json" ] && nl_say "  $slug $t: present" || { nl_say "  $slug $t: RUN (think-on, max_tokens $THINK_MAX)"; RUNS=$((RUNS+1)); }; done
done
DL=$(deadline_epoch); nl_say "  planned RUN: $RUNS (hard stop $(date -u -d "@$DL" "+%H:%M") UTC, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run"; exit 2; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(deadline_epoch)
nl_say "draining Donald (proxy: stop kills the child too)"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"

# (1) gemma 64K rerun, logged
art=$RIG/results/gemma-4-31b-q4-0-it/longcontext-64k-rerun/longcontext_detail.json
if [ ! -f "$art" ]; then
  t0=$(date +%s); nl_say "=== gemma64k rerun start ==="
  mkdir -p "$(dirname "$art")"
  timeout 1800 $PY - "$GEMMA" >> ~/secondtier-run.log 2>&1 <<'PYEOF'
import sys, json
from pathlib import Path
from lib.config import get
from lib.agentic.base import load_system_prompt
from lib.agentic.longcontext import LongContextUseEval
from lib.evals.base import LLMClient
from lib.quality import start_llama_server, stop_llama_server
path = sys.argv[1]
lc = get("agentic.evals")["longcontext"]
sp = load_system_prompt(get("agentic.system_prompt"))
rd = Path("results/gemma-4-31b-q4-0-it/longcontext-64k-rerun"); rd.mkdir(parents=True, exist_ok=True)
depth = 65536; need = depth + lc["max_tokens"] + 2048
print(f"[gemma64k] starting server ctx_size={need}", flush=True)
proc = start_llama_server(path, ctx_size=need)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=lc["think"], timeout=600) as c:
        r = LongContextUseEval(c, sp, Path(get("agentic.data_dir")), depths=[depth], limit=lc["limit"], results_dir=rd, max_tokens=lc["max_tokens"]).evaluate()
    print("[gemma64k] result:", json.dumps(r), flush=True)
    prog = json.load(open(rd / "longcontext_progress.json"))
    print("[gemma64k] errors:", json.dumps(prog.get("errors", {}))[:2000], flush=True)
finally:
    stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  if nl_judge_artifact gemma64k $rc "$art" "$t0" gemma64k; then nl_say "  gemma64k: $(grep -a '\[gemma64k\] result' ~/secondtier-run.log | tail -1 | cut -c1-160)"; grep -a '\[gemma64k\] errors' ~/secondtier-run.log | tail -1 | cut -c1-400 >> ~/secondtier.log; else nl_say "  gemma64k: FAILED rc=$rc"; fi
fi

# (2) second-tier legs
for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || continue
  todo=""; for t in $TASKS; do [ -f "$RIG/results/$slug-thinkon/${t}_detail.json" ] || todo="$todo $t"; done
  [ -n "$todo" ] || { for t in $TASKS; do nl_mark "${slug}_$t" skipped pre-existing; done; continue; }
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 1800 ] || { nl_say "window exhausted before $slug ($left s left), remaining rows resumable"; for t in $todo; do nl_mark "${slug}_$t" failed "window exhausted (resumable)"; done; continue; }
  t0=$(date +%s); nl_say "=== $slug second-tier THINK-ON:$todo (budget $((left/60)) min) ==="
  timeout "$left" $PY -u - "$path" "$slug" "$THINK_MAX" $todo >> ~/secondtier-run.log 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
path, slug, think_max, tasks = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
rd = Path("results") / f"{slug}-thinkon"; rd.mkdir(parents=True, exist_ok=True)
proc = start_llama_server(path, ctx_size=24576)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=1800) as client:
        for t in tasks:
            t0 = time.time()
            if t == "ifeval": ev = IFEvalEval(client=client, results_dir=rd, max_tokens=think_max)
            elif t == "math500": ev = Math500Eval(client=client, results_dir=rd, max_tokens=think_max)
            else: ev = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=think_max)
            r = ev.evaluate()
            r["regime"] = f"think-on, max_tokens {think_max}, ctx 24576, zero-shot, greedy"
            (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
            print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  for t in $todo; do a="$RIG/results/$slug-thinkon/${t}_detail.json"
    if [ -f "$a" ] && [ "$(stat -c %Y "$a")" -ge "$t0" ]; then nl_mark "${slug}_$t" done "$(grep -a "\[$slug\] $t:" ~/secondtier-run.log | tail -1 | cut -c1-120)"; nl_say "  $slug $t: $(grep -a "\[$slug\] $t:" ~/secondtier-run.log | tail -1 | sed "s/.*$t: //" | cut -c1-100)"
    else nl_mark "${slug}_$t" failed "rc=$rc no fresh artifact"; nl_say "  $slug $t: FAILED rc=$rc"; fi
  done
done
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || nl_donald_restore
remaining=0; for row in $ROWS; do slug="${row%%:*}"; for t in $TASKS; do [ -f "$RIG/results/$slug-thinkon/${t}_detail.json" ] || remaining=$((remaining+1)); done; done
nl_say "DONE rc=0 manifest $MANIFEST ($remaining legs still resumable)"
[ "$remaining" -eq 0 ] || exit 3
