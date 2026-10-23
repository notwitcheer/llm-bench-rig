#!/bin/bash
# bonsai-probe.sh (2026-10-08, mercury, operator "go"): sizing probe for Ternary-Bonsai-2-27B PTQ1_0 on the second-tier
# think-on board eval (same classes, same regime: think-on, max_tokens 16384, ctx 24576, greedy), PrismML llama.cpp fork
# (~/llama.cpp-prism, prism 4fbda12) swapped in for llama_cpp.server_bin. math500 first 5, ifeval/humaneval_plus/mbpp_plus first 3.
# short (<30 min), allowed outside heures creuses. Donald drained, restored on EXIT. own results dir, never the board dirs.
set -uo pipefail
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
GGUF=${GGUF:-/home/witcheer/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf}
OUT=${OUT:-results/probe-bonsai2-27b-ptq1-0-2026-10-08}
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 3; sudo -n systemctl start llama-server.service; echo "== Donald restore rc=$? $(date)"; }
trap restore EXIT
echo "== probe start $(date) $GGUF"
sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && break; sleep 2; done
echo "drain ${v}MiB"
cd $RIG; export PYTHONPATH=$RIG
timeout 1800 $PY -u - "$GGUF" "$OUT" <<'PYEOF'
import sys, json, time
from pathlib import Path
import lib.quality as q
_orig = q.get
q.get = lambda k, *a, **kw: "/home/witcheer/llama.cpp-prism/build/bin/llama-server" if k == "llama_cpp.server_bin" else _orig(k, *a, **kw)
print("server cmd:", " ".join(q.build_server_command(sys.argv[1], port=8090, ngl=99, ctx_size=24576)), flush=True)
from lib.evals.base import LLMClient
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
path, rd = sys.argv[1], Path(sys.argv[2]); rd.mkdir(parents=True, exist_ok=True)
t0 = time.time(); proc = q.start_llama_server(path, ctx_size=24576); print(f"server up in {time.time()-t0:.0f}s", flush=True)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=1800) as c:
        for t, n in (("math500", 5), ("ifeval", 3), ("humaneval_plus", 3), ("mbpp_plus", 3)):
            t1 = time.time()
            if t == "ifeval": ev = IFEvalEval(client=c, results_dir=rd, max_tokens=16384, limit=n)
            elif t == "math500": ev = Math500Eval(client=c, results_dir=rd, max_tokens=16384, limit=n)
            else: ev = EvalPlusEval(client=c, variant=t, results_dir=rd, max_tokens=16384, limit=n)
            r = ev.evaluate()
            (rd / f"{t}_detail.json").write_text(json.dumps(r, indent=2))
            print(f"[probe] {t} n={n}: {json.dumps({k: r.get(k) for k in ('score','correct','total','parse_failures','capped_rate','tokens_per_correct')})} in {time.time()-t1:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
echo "== eval rc=$? $(date)"
grep -a "tokens/s\|eval time" ~/bench-server.log | tail -4
echo PROBE_END
