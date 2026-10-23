#!/bin/bash
# 50-matharena-fresh.sh: idle-queue item (heures creuses runner). the "fresh 2026 maths vs the old test" study
# (operator 2026-09-28, GPU thread: "I like fresh 2026 maths vs old test yes").
# question: the ten models of the 16k think-on board score 72 to 95 on MATH-500 (2021, all over training data). what do
# they score on maths published AFTER they were released, and on the same kind of maths published BEFORE?
# sets (MathArena, lib/evals/matharena.py; Hub createdAt in brackets):
#   fresh:  ArXivMath 08/26 (57 research-level questions from arXiv papers submitted in Aug 2026, published 2026-09-13).
#           every board model was released before it (latest: Qwopus3.8-27B-Flash 2026-09-03, a Qwen3.8 fine-tune).
#   control, same format, old: ArXivMath 12/25 (17), 01/26 (23), 02/26 (32), published 2026-02-16 to 2026-03-02,
#           before every board model's release (earliest: Gemma 4 31B 2026-03-11).
#   this year's contests: AIME 2026 (30) + HMMT Feb 2026 (33), published Feb 2026 (also before every release).
# regime: think-on, max_tokens 32768, ctx 40960, zero-shot, greedy, one pass (the rig standard, same as the 32k MATH-500 pass).
# loop is TASK-MAJOR: every row gets the headline set (arxivmath-0826) before any row starts the controls, so a partial
# study already carries the headline. artefacts results/<slug>-matharena-32k/<task>_detail.json, per-item progress saved
# after every item (a hard stop loses the in-flight question only). exit 3 while legs remain, 0 when all 60 legs landed.
# grader: strict (Qwen2.5-Math grade + string equality); MathArena grades ArXivMath with an LLM judge, so agreement with
# their labels is measured before any score is quoted (scripts/matharena_grader_agreement.py); predicted/expected kept
# per item for an offline re-grade.
# probe 2026-09-28 08:1x CEST (~/matharena-probe.sh, Qwen3.8-27B Q6_K, first 3 arxivmath-0826 items): see the vault status note.
set -uo pipefail
NL_TAG=matharena
MANIFEST=/home/witcheer/matharena-status.json
THINK_MAX=32768
CTX=40960
SUFFIX=matharena-32k
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PORT=8090
TASKS="ma_arxivmath-0826 ma_arxivmath-1225 ma_arxivmath-0126 ma_arxivmath-0226 ma_aime_2026 ma_hmmt_feb_2026"
ROWS="
qwen3-8-27b-q6-k:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf
qwen3-6-35b-a3b-ud-q5-k-m:/home/witcheer/models/qwen36-35b-base/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m:/home/witcheer/models/nemotron-35-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
ornith-1-5-35b-q4-k-m:/home/witcheer/models/ornith-15-35b/Ornith-1.5-35B-Q4_K_M.gguf
gemma-4-31b-q4-0-it:/home/witcheer/models/gemma4-qat-q4/gemma-4-31B_q4_0-it.gguf
qwen3-6-27b-q6-k:/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf
qwable-27b-q4-k-m:/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf
qwopus3-8-27b-flash-q6-k:/home/witcheer/models/qwopus38-flash/Qwopus3.8-27B-Flash-MTP-Q6_K.gguf
qwen3-8-27b-ud-iq3-xxs:/home/witcheer/models/qwen38-27b/Qwen3.8-27B-UD-IQ3_XXS.gguf
ornith-9b:/home/witcheer/models/ornith-15-9b/Ornith-1.5-9B-Q6_K.gguf
"
deadline_epoch() { nl_deadline_epoch; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) budget $THINK_MAX ctx $CTX suffix -$SUFFIX"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/matharena.py" "$RIG/lib/evals/math500.py"
nl_check_disk 50
$PY -c "import sys; sys.path.insert(0,'$RIG'); from lib.qwen_math_grader import grade; assert grade(r'\frac{1}{2}','0.5')" >/dev/null 2>&1 && nl_say "  ok: grader import (frac 1/2 == 0.5)" || { nl_say "  grader import FAILED (would under-count silently)"; exit 2; }
$PY -c "
from datasets import load_dataset
want={'arxivmath-0826':57,'arxivmath-1225':17,'arxivmath-0126':23,'arxivmath-0226':32,'aime_2026':30,'hmmt_feb_2026':33}
for k,n in want.items(): assert len(load_dataset('MathArena/'+k,split='train'))==n,k
" >/dev/null 2>&1 && nl_say "  ok: six MathArena sets (192 items)" || { nl_say "  dataset probe FAILED"; exit 2; }
PH=""; RUNS=0
for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; continue; }
  PH="$PH ${slug}_$t"; [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] && nl_say "  $slug $t: present" || RUNS=$((RUNS+1)); done
done
DL=$(deadline_epoch); nl_say "  planned RUN: $RUNS legs (hard stop $(date -u -d "@$DL" "+%H:%M") UTC, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run"; exit 2; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(deadline_epoch)
nl_say "draining Donald (proxy: stop kills the child too)"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"

for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || continue
  a="$RIG/results/$slug-$SUFFIX/${t}_detail.json"
  [ -f "$a" ] && { nl_mark "${slug}_$t" skipped pre-existing; continue; }
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 900 ] || { nl_say "window exhausted before $slug $t ($left s left), resumable"; nl_mark "${slug}_$t" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_say "=== $slug $t think-on 32k (budget $((left/60)) min) ==="
  timeout "$left" $PY - "$path" "$slug" "$THINK_MAX" "$CTX" "$SUFFIX" "$t" >> ~/matharena-run.log 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.matharena import MathArenaEval
path, slug, think_max, ctx, suffix, t = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
rd = Path("results") / f"{slug}-{suffix}"; rd.mkdir(parents=True, exist_ok=True)
proc = start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=3600) as client:
        t0 = time.time()
        r = MathArenaEval(client=client, task=t, results_dir=rd, max_tokens=think_max).evaluate()
        r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
        r["bench"] = "matharena-fresh-32k (fresh vs old maths study, 2026-09-28)"
        (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  if [ -f "$a" ] && [ "$(stat -c %Y "$a")" -ge "$t0" ]; then nl_mark "${slug}_$t" done "$(grep -a "\[$slug\] $t:" ~/matharena-run.log | tail -1 | cut -c1-120)"; nl_say "  $slug $t: $(grep -a "\[$slug\] $t:" ~/matharena-run.log | tail -1 | sed "s/.*$t: //" | cut -c1-100)"
  else nl_mark "${slug}_$t" failed "rc=$rc no fresh artifact"; nl_say "  $slug $t: FAILED rc=$rc"; fi
done; done
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || nl_donald_restore
remaining=0; for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] || remaining=$((remaining+1)); done; done
nl_say "DONE rc=0 manifest $MANIFEST ($remaining legs still resumable)"
[ "$remaining" -eq 0 ] || exit 3
