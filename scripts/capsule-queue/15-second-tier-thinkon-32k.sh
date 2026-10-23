#!/bin/bash
# 15-second-tier-thinkon-32k.sh: idle-queue item (heures creuses runner). SECOND full second-tier think-on pass at a 32k
# 2026-09-15: qwable-27b q4_k_m added (operator yes): 16k pass capping 27% at 150/500 math500, same shape as the two rows above.
# completion budget, kept as a SEPARATE bench from the 16k pass (operator 2026-09-13: "second full pass with 32k, separated").
# why: at 16k the maths rows hit the budget hard (qwen3-6-27b math500 27% capped, qwen3-6-35b 30%), so those scores are floors.
# artifacts: results/<slug>-thinkon-32k/<task>_detail.json (fresh dirs, never touches the -thinkon 16k dirs), regime stamped 32768.
# ctx 40960 = 32768 completion + prompt headroom (second-tier prompts are < 2k tokens). a server that cannot fit the KV cache fails
# start_llama_server cleanly and the row is marked failed, nothing else is harmed.
# exits 3 while legs remain (stays queued, resumable), 0 once every leg has a fresh artifact (runner moves it to done/).
# NARROWED 2026-09-14 (operator: "following your lead"): only the legs that capped above 10% at 16k, i.e. math500 on
# qwen3-6-27b (27%) and qwen3-6-35b-a3b (30%). every other 16k leg capped under 7%, a 32k rerun there buys nothing.
# the 10-row/40-leg version is kept at ~/15-second-tier-thinkon-32k.sh.bak-10rows-2026-09-14.
# same night-lib contract as 10-second-tier-thinkon.sh: drains Donald, restores on EXIT, hard stop = end of the heures creuses block.
set -uo pipefail
NL_TAG=secondtier32k
MANIFEST=/home/witcheer/secondtier32k-status.json
THINK_MAX=32768
CTX=40960
SUFFIX=thinkon-32k
export MANIFEST
. /home/witcheer/night-lib.sh
# WEEKEND-ONLY + sorted after 40 (operator yes 2026-09-22, GPU thread): at ~3.4 min/row the remaining legs need ~45 h of card
# time; on alternate weeknight blocks that is three weeks with 20/30/40 starved. weeknight blocks go to 20/30/40, this item runs
# only when the local day is Sat/Sun (the EDF week-end block, whole days). exit 3 = stays queued, resumable, drain moves on.
# renamed 15- -> 45- the same day; log is now ~/idleq-45-second-tier-thinkon-32k.log, manifest + results dirs unchanged so resume holds.
# 2026-09-23 (operator yes, GPU thread): renamed 45- -> 15- so the 32k legs own the weekend block ahead of 20/30/40 (~45 h of card time vs ~47 h weekend); weekday guard unchanged. log is now ~/idleq-15-second-tier-thinkon-32k.log; backup ~/45-second-tier-thinkon-32k.sh.bak-weekendfirst-2026-09-23.
if [ "${DRYRUN:-0}" != 1 ] && [ "$(nl_hc_dow)" -lt 6 ]; then
  nl_say "lane: weekend-only item, weekday block left to 20/30/40 (exit 3, resumable)"; exit 3
fi
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
PORT=8090
TASKS="math500"
ROWS="
qwen3-6-35b-a3b-ud-q5-k-m:/home/witcheer/models/qwen36-35b-base/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
qwen3-6-27b-q6-k:/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf
qwable-27b-q4-k-m:/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf
"
deadline_epoch() { nl_deadline_epoch; }

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) budget $THINK_MAX ctx $CTX suffix -$SUFFIX"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py"
nl_check_disk 50
$PY -c "import sys; sys.path.insert(0,'$RIG'); from lib.quality import SECOND_TIER_TASKS; assert {'ifeval','math500','humaneval_plus','mbpp_plus'} <= SECOND_TIER_TASKS" && nl_say "  ok: second-tier tasks registered" || { nl_say "  second-tier tasks missing"; exit 2; }
$PY -c "from datasets import load_dataset; assert len(load_dataset('google/IFEval',split='train'))==541; assert len(load_dataset('HuggingFaceH4/MATH-500',split='test'))==500; assert len(load_dataset('evalplus/humanevalplus',split='test'))==164; assert len(load_dataset('evalplus/mbppplus',split='test'))==378" >/dev/null 2>&1 && nl_say "  ok: four datasets" || { nl_say "  dataset probe FAILED"; exit 2; }
PH=""; RUNS=0
for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; continue; }
  for t in $TASKS; do PH="$PH ${slug}_$t"; [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] && nl_say "  $slug $t: present" || { nl_say "  $slug $t: RUN (think-on, max_tokens $THINK_MAX)"; RUNS=$((RUNS+1)); }; done
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

for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || continue
  todo=""; for t in $TASKS; do [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] || todo="$todo $t"; done
  [ -n "$todo" ] || { for t in $TASKS; do nl_mark "${slug}_$t" skipped pre-existing; done; continue; }
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 1800 ] || { nl_say "window exhausted before $slug ($left s left), remaining rows resumable"; for t in $todo; do nl_mark "${slug}_$t" failed "window exhausted (resumable)"; done; continue; }
  t0=$(date +%s); nl_say "=== $slug second-tier THINK-ON 32k:$todo (budget $((left/60)) min) ==="
  timeout "$left" $PY - "$path" "$slug" "$THINK_MAX" "$CTX" "$SUFFIX" $todo >> ~/secondtier32k-run.log 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
path, slug, think_max, ctx, suffix, tasks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6:]
rd = Path("results") / f"{slug}-{suffix}"; rd.mkdir(parents=True, exist_ok=True)
proc = start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=3600) as client:
        for t in tasks:
            t0 = time.time()
            if t == "ifeval": ev = IFEvalEval(client=client, results_dir=rd, max_tokens=think_max)
            elif t == "math500": ev = Math500Eval(client=client, results_dir=rd, max_tokens=think_max)
            else: ev = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=think_max)
            r = ev.evaluate()
            r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
            r["bench"] = "second-tier-32k (separate pass, not comparable to the 16k -thinkon rows)"
            (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
            print(f"[{slug}] {t}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    stop_llama_server(proc)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  for t in $todo; do a="$RIG/results/$slug-$SUFFIX/${t}_detail.json"
    if [ -f "$a" ] && [ "$(stat -c %Y "$a")" -ge "$t0" ]; then nl_mark "${slug}_$t" done "$(grep -a "\[$slug\] $t:" ~/secondtier32k-run.log | tail -1 | cut -c1-120)"; nl_say "  $slug $t: $(grep -a "\[$slug\] $t:" ~/secondtier32k-run.log | tail -1 | sed "s/.*$t: //" | cut -c1-100)"
    else nl_mark "${slug}_$t" failed "rc=$rc no fresh artifact"; nl_say "  $slug $t: FAILED rc=$rc"; fi
  done
done
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || nl_donald_restore
remaining=0; for row in $ROWS; do slug="${row%%:*}"; for t in $TASKS; do [ -f "$RIG/results/$slug-$SUFFIX/${t}_detail.json" ] || remaining=$((remaining+1)); done; done
nl_say "DONE rc=0 manifest $MANIFEST ($remaining legs still resumable)"
[ "$remaining" -eq 0 ] || exit 3
