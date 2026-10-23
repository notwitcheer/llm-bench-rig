#!/bin/bash
# matharena-probe-contest.sh: pace + mechanics probe for the contest pivot (2026-10-01, operator "yes" in the GPU thread).
# Qwen3.8-27B Q6_K (Donald's served model, the slowest dense row of the 5), AIME 2026 first 3 + HMMT Feb 2026 first 2,
# think-on 32k, ctx 40960, greedy. results into results/probe-matharena-contest (never read by the study skip).
# Donald drained, restored on EXIT. waits for the 08:00 brief to finish first.
set -uo pipefail
exec > >(tee -a /home/witcheer/matharena-probe-contest.log) 2>&1
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
now=$(date +%s); go=$(date -d "today 08:07" +%s); [ "$now" -lt "$go" ] && { echo "sleeping $((go-now)) s until 08:07 (after Donald's brief)"; sleep $((go-now)); }
echo "start $(date)"
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 3; echo "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { echo "drain clear ${v}MiB"; break; }; sleep 2; done
cd $RIG; export PYTHONPATH=$RIG
timeout 1800 $PY - <<'PYEOF'
import json, time
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.matharena import MathArenaEval
path = "/home/witcheer/models/qwen38-27b/Qwen3.8-27B-Q6_K.gguf"
rd = Path("results/probe-matharena-contest"); rd.mkdir(parents=True, exist_ok=True)
t0 = time.time(); proc = start_llama_server(path, ctx_size=40960); print("server up in", round(time.time()-t0), "s", flush=True)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=3600) as c:
        for task, n in (("ma_aime_2026", 3), ("ma_hmmt_feb_2026", 2)):
            t1 = time.time()
            r = MathArenaEval(c, task, limit=n, results_dir=rd, max_tokens=32768).evaluate()
            print("PROBE", task, json.dumps({k: r.get(k) for k in ("score","correct","total","parse_failures","capped_rate","tokens_per_correct","completion_tokens_median")}), f"{time.time()-t1:.0f}s", flush=True)
finally:
    stop_llama_server(proc)
PYEOF
echo "probe rc=$? $(date)"
