#!/bin/bash
# 58-matharena-contest-ext.sh: idle-queue item. EXTENSION of 57 (2026-10-04 07:2x CEST, queued by mercury after the
# operator asked "why is the GPU idle?": the weekend cheap block had ~16.5 h left and an empty queue).
# rows added: the two dense 27B rows whose 32k MATH-500 pass now clears the same "under ~12% capped" bar 57 used:
#   Qwen3.6-27B Q6_K 92.0 (32k, cap 7.8%), Qwable 27B Q4_K_M 90.0 (32k, cap 10.2%) (results/<slug>-thinkon-32k).
# body, regime, grader, veto and suffix identical to 57; only tag, manifest, run log, rows and bench string change.
# original 57 header follows.
# 57-matharena-contest.sh: idle-queue item (heures creuses runner). the contest pivot of the "fresh maths vs the old
# test" study (operator 2026-10-01 GPU thread "yes", after the ArXivMath pilot verdict).
# why the pivot: Qwen3.6-35B-A3B on ArXivMath 08/26 (57) scored 0/57 at 32k (55 capped) and 1/54 at 64k (34 capped,
# the 17 new finishers all wrong), so research-level maths is a "they don't finish" line, not a headline.
# question now: five board models score 88 to 95 on MATH-500 (2021). what do they score on this year's contests?
# sets (MathArena, lib/evals/matharena.py): AIME 2026 (30) + HMMT Feb 2026 (33), published Feb 2026 (Hub createdAt
# 2026-02-13 / 02-19), i.e. "this year's contests", NOT unseen: every board model was released after them.
# rows = the five with a MATH-500 cell under ~12% capped (the comparison only means something against a real score):
#   Qwen3.8-27B Q6_K 95.2 (16k, cap 1.6%, Donald's served model), Gemma 4 31B 94.0 (16k, 2.2%), Qwen3.6-35B-A3B 91.2
#   (32k pass, 12%), Nemotron 3.5 Lightning 90.2 (16k, 6.2%), Ornith 1.5 35B 88.2 (16k, 2.2%).
# regime: think-on, max_tokens 32768, ctx 40960, zero-shot, greedy, one pass. task-major: AIME for every row first.
# artefacts results/<slug>-matharena-32k/<task>_detail.json (same suffix as the pilot, different task files).
# body = 50-matharena-fresh.sh (staged, never run) via edits: tag, manifest, run log, tasks, rows, dataset probe, bench.
# the body ran for real as 50-matharena-pilot (hard stop + resume on 09-30, COMPLETE 10-01) and 55 (rc=124 + resume).
# probe 2026-10-01 08:07 CEST (~/matharena-probe-contest.sh, Qwen3.8-27B Q6_K, AIME 2026 x3 + HMMT x2): see the
# vault status note 99-personal/benchmark-rig/matharena-fresh-status.md.
# grader: exact_veto=True (lib/evals/matharena.py, 2026-10-01): the tolerant numeric compare credited 46 wrong answers on
# MathArena's HMMT Feb 2026 outputs (idx 31 gold 3^{2025}, idx 9 gold 50(1-1/(2^101-1))); veto -> 0 false positives,
# HMMT agreement 99.6%, AIME 99.7% unchanged. the veto is off by default, so 50-/55- legs grade as before.
# PLAN checks the veto itself (latex2sympy2_extended import + the HMMT idx 9 case) so a missing dependency cannot
# silently turn it into a no-op.
set -uo pipefail
NL_TAG=matharena-contest-ext
MANIFEST=/home/witcheer/matharena-contest-ext-status.json
THINK_MAX=32768
CTX=40960
SUFFIX=matharena-32k
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PORT=8090
TASKS="ma_aime_2026 ma_hmmt_feb_2026"
ROWS="
qwen3-6-27b-q6-k:/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf
qwable-27b-q4-k-m:/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf
"
deadline_epoch() { nl_deadline_epoch; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) budget $THINK_MAX ctx $CTX suffix -$SUFFIX"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/matharena.py" "$RIG/lib/evals/math500.py"
nl_check_disk 50
$PY -c "import sys; sys.path.insert(0,'$RIG'); from lib.qwen_math_grader import grade; assert grade(r'\frac{1}{2}','0.5')" >/dev/null 2>&1 && nl_say "  ok: grader import (frac 1/2 == 0.5)" || { nl_say "  grader import FAILED (would under-count silently)"; exit 2; }
$PY -c "import sys; sys.path.insert(0,'$RIG'); from latex2sympy2_extended import latex2sympy; from lib.evals.matharena import grade_with_timeout as g; assert g('50', r'50(1 - \\frac{1}{2^{101} - 1})', veto=True, return_vetoed=True)[2]; assert g('277','277', veto=True)[0]" >/dev/null 2>&1 && nl_say "  ok: exact-number veto (50 vs 50(1-1/(2^101-1)) vetoed, 277 kept)" || { nl_say "  exact-number veto check FAILED"; exit 2; }
$PY -c "
from datasets import load_dataset
want={'aime_2026':30,'hmmt_feb_2026':33}
for k,n in want.items(): assert len(load_dataset('MathArena/'+k,split='train'))==n,k
" >/dev/null 2>&1 && nl_say "  ok: two MathArena contest sets (63 items)" || { nl_say "  dataset probe FAILED"; exit 2; }
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
  timeout "$left" $PY - "$path" "$slug" "$THINK_MAX" "$CTX" "$SUFFIX" "$t" >> ~/matharena-contest-ext-run.log 2>&1 <<'PYEOF'
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
        r = MathArenaEval(client=client, task=t, results_dir=rd, max_tokens=think_max, exact_veto=True).evaluate()
        r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
        r["bench"] = "matharena-contest-32k (this year's contests vs MATH-500, 2026-10-01; 27B rows added 2026-10-04)"
        (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  if [ -f "$a" ] && [ "$(stat -c %Y "$a")" -ge "$t0" ]; then nl_mark "${slug}_$t" done "$(grep -a "\[$slug\] $t:" ~/matharena-contest-ext-run.log | tail -1 | cut -c1-120)"; nl_say "  $slug $t: $(grep -a "\[$slug\] $t:" ~/matharena-contest-ext-run.log | tail -1 | sed "s/.*$t: //" | cut -c1-100)"
  else nl_mark "${slug}_$t" failed "rc=$rc no fresh artifact"; nl_say "  $slug $t: FAILED rc=$rc"; fi
done; done
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || nl_donald_restore
remaining=0; for t in $TASKS; do for row in $ROWS; do slug="${row%%:*}"; [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] || remaining=$((remaining+1)); done; done
nl_say "DONE rc=0 manifest $MANIFEST ($remaining legs still resumable)"
[ "$remaining" -eq 0 ] || exit 3
