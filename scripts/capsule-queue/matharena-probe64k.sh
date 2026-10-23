#!/bin/bash
# matharena-probe64k.sh: does ArXivMath 08/26 land with more room? (operator "yes following your leads", 2026-09-30, GPU thread)
# the 32k pilot capped 18/18 (Qwen3.6-35B-A3B UD-Q5_K_M, ~134 s per item, still mid-derivation at the cap).
# this probe: same model, same first 5 items (idx 0-4, all capped at 32k), think-on 65536, ctx 73728, greedy.
# short probe (hard timeout 1800 s incl. load), so it runs outside heures creuses by the electricity rule.
# results into results/probe-matharena-64k (separate dir, never read by the study's exact-artefact skip).
# Donald drained, restored on EXIT.
set -uo pipefail
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 3; echo "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
echo "start $(date '+%F %T %Z')"
sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 60); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 4000 ] && { echo "drain clear ${v}MiB"; break; }; sleep 2; done
cd $RIG; export PYTHONPATH=$RIG
timeout 1800 $PY - <<'PYEOF'
import json, time
from pathlib import Path
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.evals.matharena import MathArenaEval
path = "/home/witcheer/models/qwen36-35b-base/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
rd = Path("results/probe-matharena-64k"); rd.mkdir(parents=True, exist_ok=True)
t0 = time.time(); proc = start_llama_server(path, ctx_size=73728); print("server up in", round(time.time()-t0), "s", flush=True)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=True, timeout=3600) as c:
        r = MathArenaEval(c, "ma_arxivmath-0826", limit=5, results_dir=rd, max_tokens=65536).evaluate()
        r["regime"] = "think-on, max_tokens 65536, ctx 73728, zero-shot, greedy (probe, first 5 items)"
        (rd / "ma_arxivmath-0826_detail.json").write_text(json.dumps(r, indent=2))
        print(json.dumps({k: r.get(k) for k in ("score","correct","total","parse_failures","capped_rate","tokens_per_correct","completion_tokens_median")}), flush=True)
finally:
    stop_llama_server(proc)
PYEOF
echo "probe rc=$? end $(date '+%F %T %Z')"
