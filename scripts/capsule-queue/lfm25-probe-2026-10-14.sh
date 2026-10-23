#!/bin/bash
# lfm-probe.sh: GPU sizing probe for LFM2.5-24B-A2B (2026-10-14, mercury subagent). NOT a queue item.
# think-on second-tier regime of the comparator row (Qwen3.6-35B-A3B UD-Q5_K_M -thinkon): max_tokens 16384, ctx 24576,
# zero-shot, greedy, harness llama.cpp-nl build. first 5 items of each task: math500 humaneval_plus mbpp_plus ifeval gpqa.
# plus llama-bench pp512/tg128 at depth 0 and 8192 on the downloaded quants. Donald drained, trap restore EXIT.
# waits (until 12:55 local at the latest) for the card to be clear: < 3 GB used and no foreign GPU process.
set -uo pipefail
NL_TAG=lfmprobe
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
BIN=/home/witcheer/llama.cpp-nl/build/bin
Q5=/home/witcheer/models/lfm25-24b/LFM2.5-24B-A2B-UD-Q5_K_XL.gguf
Q4=/home/witcheer/models/lfm25-24b/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf
Q8=/home/witcheer/models/lfm25-24b-liquid/LFM2.5-24B-A2B-Q8_0.gguf
SLUG=probe-lfm2-5-24b-a2b-ud-q5-k-xl-thinkon
LIMIT=${LIMIT:-5}
HARD=$(date -d "today 13:40" +%s)   # booked block 14:10 CEST; probe must be over by 13:50
nl_say "PLAN: probe $SLUG limit=$LIMIT, hard stop $(date -d @$HARD +%H:%M) local"
nl_check_path "$PY" "$BIN/llama-server" "$BIN/llama-bench" "$Q5" "$Q4"
nl_gate
# card-clear wait
WAIT_UNTIL=$(date -d "today 12:55" +%s)   # latest start so the probe ends before 13:40
for i in $(seq 1 100000); do
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v "^$" | wc -l)
  foreign=$(tmux ls 2>/dev/null | grep -E "^(pushchk|probe-|idleq|night|secondtier|gpqa|recipe)" | wc -l)
  if [ "${v:-99999}" -lt 3000 ] && [ "$apps" -eq 0 ] && [ "$foreign" -eq 0 ]; then nl_say "card clear (${v} MiB, $apps apps, $foreign foreign tmux)"; break; fi
  [ $((i % 10)) -eq 1 ] && nl_say "waiting for the card (${v} MiB, $apps apps, $foreign foreign tmux)"; sleep 30
  [ "$(date +%s)" -ge "$WAIT_UNTIL" ] && { nl_say "ABORT: card never cleared before $(date -d @$WAIT_UNTIL +%H:%M)"; exit 2; }
done
restore() { pkill -9 -x llama-server 2>/dev/null; pkill -9 -x llama-bench 2>/dev/null; kill "${SAMP:-0}" 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; sleep 5; curl -s localhost:8090/health | grep -qi ok && nl_say "Donald proxy OK" || nl_say "Donald proxy NOT ok"; }
trap restore EXIT
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; done
# VRAM sampler
SAMPF=/home/witcheer/lfm-probe-vram.csv; : > $SAMPF
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits | head -1)" >> $SAMPF; sleep 2; done ) &
SAMP=$!
# 1) speed: llama-bench per quant
[ "${SKIP_BENCH:-0}" = 1 ] && nl_say "SKIP_BENCH=1: no llama-bench pass"
for g in $( [ "${SKIP_BENCH:-0}" = 1 ] || echo "$Q5 $Q4 $Q8" ); do
  [ -f "$g" ] || { nl_say "skip bench (missing) $g"; continue; }
  nl_say "llama-bench $(basename $g)"
  timeout 600 $BIN/llama-bench -m "$g" -ngl 99 -fa on -p 512 -n 128 -d 0,8192 -r 2 -o json > /home/witcheer/lfm-bench-$(basename $g .gguf).json 2>/home/witcheer/lfm-bench-$(basename $g .gguf).err
  python3 -c "import json,sys;[print('   ',r.get('n_prompt'),r.get('n_gen'),'d',r.get('n_depth'),round(r.get('avg_ts',0),1),'tok/s') for r in json.load(open(sys.argv[1]))]" /home/witcheer/lfm-bench-$(basename $g .gguf).json || nl_say "  bench parse failed"
done
# 2) eval probe, one server, five tasks
left=$(( HARD - $(date +%s) )); [ "$left" -gt 2400 ] && left=2400
EVT0=$(date +%s); nl_say "eval probe start (timeout $left s)"
cd "$RIG"; export PYTHONPATH="$RIG"
timeout "$left" $PY -u - "$Q5" "$SLUG" "$LIMIT" "${PTASKS:-math500 humaneval_plus mbpp_plus ifeval gpqa}" >> /home/witcheer/lfm-probe-run.log 2>&1 <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
from lib.evals.base import LLMClient
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
from lib.evals.gpqa import GPQAEval
path, slug, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
M = 16384
rd = Path("results") / slug; rd.mkdir(parents=True, exist_ok=True)
print(f"[{slug}] server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=24576))}", flush=True)
t0 = time.time(); proc = q.start_llama_server(path, ctx_size=24576); print(f"[{slug}] server up in {time.time()-t0:.0f}s", flush=True)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=1800) as client:
        for t in sys.argv[4].split():
            t0 = time.time()
            if t == "ifeval": ev = IFEvalEval(client=client, results_dir=rd, max_tokens=M, limit=limit)
            elif t == "math500": ev = Math500Eval(client=client, results_dir=rd, max_tokens=M, limit=limit)
            elif t == "gpqa": ev = GPQAEval(client, results_dir=rd, max_tokens=M, limit=limit)
            else: ev = EvalPlusEval(client=client, variant=t, results_dir=rd, max_tokens=M, limit=limit)
            r = ev.evaluate(); wall = time.time() - t0
            r["regime"] = f"think-on, max_tokens {M}, ctx 24576, zero-shot, greedy (PROBE limit {limit})"
            (rd / f"{t}_probe.json").write_text(json.dumps(r, indent=2, default=str))
            print(f"[{slug}] {t}: " + json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','capped_rate','completion_tokens_total','completion_tokens_median','reasoning_fallback_count')}) + f" wall {wall:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
rc=$?
nl_say "eval probe rc=$rc"
cp /home/witcheer/bench-server.log /home/witcheer/lfm-probe-server.log 2>/dev/null
kill $SAMP 2>/dev/null
python3 -c "
import csv
rows=[r for r in csv.reader(open('$SAMPF')) if len(r)>=2]
m=max(int(float(r[1])) for r in rows); e=[int(float(r[1])) for r in rows if int(r[0])>=$EVT0]; print('VRAM peak MiB (whole run incl. bench)', m, '| eval-phase peak', max(e) if e else None, 'samples', len(rows))"
nl_say "DONE rc=$rc"
