#!/bin/bash
# 60-gpqa-thinkon-32k.sh: idle-queue item (heures creuses runner). SEPARATE bench: GPQA-diamond think-on at max_tokens 32768,
# ctx 40960 (same regime as the 32k MATH-500 pass), on the 40-gpqa-gaps rows whose 16k think-on leg capped above 10%
# (a score there is a floor, not a score). operator 2026-09-29 GPU thread: "following your call" on the higher-budget pass.
# rows gate themselves at PLAN time from results/<slug>-thinkon/gpqa.json: capped_rate > 0.10 -> RUN; <= 0.10 -> not needed;
# 16k json missing -> WAIT (counts as remaining, exit 3, so the Q6_K row joins once its 16k leg lands in 40-).
# 16k caps at write time: nemotron lightning 0.374 (74/198), qwable 0.227 (45/198), qwen3.6-27b Q6_K 0.147 running (11/75).
# qwen3.6-35b-a3b (0.091) and gemma 4 31b (0.025) sit under the line, so they are not rerun (hours saved: ~6 h).
# order: fastest and most-capped first. upper-bound sizing (every 16k-capped item burns the full 32k): nemotron ~2.9M tokens,
# qwable ~2.3M (~10 h at the 16k pace of ~64 tok/s), Q6_K ~1.9M (~10 h at ~52 tok/s): ~25 h worst case, less if capped items finish.
# results/<slug>-thinkon-32k/, own manifest/log, never the 16k dir. resumable per item via gpqa_progress.json. exit 3 while legs remain.
set -uo pipefail
NL_TAG=gpqa32k
MANIFEST=/home/witcheer/gpqa32k-status.json
export MANIFEST
THINK_MAX=32768
CTX=40960
SUFFIX=thinkon-32k
CAP_LINE=0.10
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
ROWS="
nvidia-nemotron-3-5-lightning-30b-a3b-q4-k-m:/home/witcheer/models/nemotron-35-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
qwable-27b-q4-k-m:/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf
qwen3-6-27b-q6-k:/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf
"
deadline_epoch() { nl_deadline_epoch; }

# row state from the 16k leg: RUN | SKIP | WAIT
row_state() { # <slug>
  local j="$RIG/results/$1-thinkon/gpqa.json"
  [ -f "$j" ] || { echo WAIT; return; }
  $PY -c "import json,sys; r=json.load(open(sys.argv[1])); c=r.get('capped_rate'); print('RUN' if c is not None and c > float(sys.argv[2]) else 'SKIP')" "$j" "$CAP_LINE"
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}) budget $THINK_MAX ctx $CTX suffix -$SUFFIX cap line $CAP_LINE"
nl_check_path "$PY" "$RIG/lib/quality.py" "$RIG/lib/evals/gpqa.py"
nl_check_disk 50
$PY -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); assert len(d)==198" >/dev/null 2>&1 && nl_say "  ok: gpqa_diamond loads (198)" || { nl_say "  gpqa dataset probe FAILED (gated grant?)"; exit 2; }
$PY -c "import inspect,sys; sys.path.insert(0,'$RIG'); from lib.evals.gpqa import GPQAEval; assert 'max_tokens' in inspect.signature(GPQAEval).parameters" && nl_say "  ok: GPQAEval(max_tokens=) available" || { nl_say "  GPQAEval lacks max_tokens (pull origin/main)"; exit 2; }
PH=""; RUNS=0; WAITS=0
for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || { nl_say "  $slug: MISSING FILE"; continue; }
  st=$(row_state "$slug")
  if [ -f "$RIG/results/$slug-$SUFFIX/gpqa.json" ]; then PH="$PH ${slug}_32k"; nl_say "  $slug 32k: present"
  elif [ "$st" = RUN ]; then PH="$PH ${slug}_32k"; nl_say "  $slug 32k: RUN (16k leg capped above $CAP_LINE)"; RUNS=$((RUNS+1))
  elif [ "$st" = WAIT ]; then PH="$PH ${slug}_32k"; nl_say "  $slug 32k: WAIT (16k think-on leg not landed yet)"; WAITS=$((WAITS+1))
  else nl_say "  $slug 32k: not needed (16k cap <= $CAP_LINE)"; fi
done
DL=$(deadline_epoch); nl_say "  planned RUN: $RUNS, waiting: $WAITS (hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away)"
if [ "$RUNS" -eq 0 ]; then [ "$WAITS" -gt 0 ] && { nl_say "only waiting rows, exit 3 (resumable)"; exit 3; }; nl_say "nothing to run"; exit 0; fi
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
DEADLINE=$(deadline_epoch)
nl_say "draining Donald (proxy: stop kills the child too)"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"

for row in $ROWS; do slug="${row%%:*}"; path="${row#*:}"; [ -f "$path" ] || continue
  art="$RIG/results/$slug-$SUFFIX/gpqa.json"; ph="${slug}_32k"
  [ -f "$art" ] && { nl_mark "$ph" skipped pre-existing; continue; }
  [ "$(row_state "$slug")" = RUN ] || continue
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 1800 ] || { nl_say "window exhausted before $ph ($left s left), resumable"; nl_mark "$ph" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_say "=== $ph gpqa think-on 32k start (budget $((left/60)) min) ==="
  timeout "$left" $PY - "$path" "$slug" "$THINK_MAX" "$CTX" "$SUFFIX" >> ~/gpqa32k-run.log 2>&1 <<'PYEOF'
import sys, json
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.gpqa import GPQAEval
path, slug, think_max, ctx, suffix = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rd = Path("results") / f"{slug}-{suffix}"; rd.mkdir(parents=True, exist_ok=True)
proc = start_llama_server(path, ctx_size=ctx)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=10800) as client:
        r = GPQAEval(client=client, results_dir=rd, max_tokens=think_max).evaluate()
finally:
    stop_llama_server(proc)
r["regime"] = f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy"
r["bench"] = "gpqa-thinkon-32k (separate higher-budget pass over rows capped >10% at 16k)"
(rd / "gpqa.json").write_text(json.dumps(r, indent=2))
print(f"[{slug}] gpqa think-on 32k: {json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','tokens_per_correct','capped_rate')})}", flush=True)
PYEOF
  rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3
  [ "$rc" = 124 ] && nl_say "  $ph: killed at the hard stop (gpqa_progress.json resumes)"
  if nl_judge_artifact "$ph" $rc "$art" "$t0" "$ph"; then nl_say "  $ph: $(grep -a "\[$slug\] gpqa think-on 32k" ~/gpqa32k-run.log | tail -1 | sed 's/.*32k: //' | cut -c1-140)"; else nl_say "  $ph: FAILED rc=$rc"; fi
done

restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || nl_donald_restore
remaining=0
for row in $ROWS; do slug="${row%%:*}"; [ -f "$RIG/results/$slug-$SUFFIX/gpqa.json" ] && continue; st=$(row_state "$slug"); [ "$st" = SKIP ] || remaining=$((remaining+1)); done
nl_say "DONE rc=0 manifest $MANIFEST ($remaining legs still resumable or waiting)"
[ "$remaining" -eq 0 ] || exit 3
