#!/bin/bash
# lfm-captail.sh: non-termination check on the GPQA items that capped at 16k in the probe (greedy, harness regime),
# plus the same items under Liquid's recommended sampling (temp 0.1, top_k 50, repeat_penalty 1.05). Donald drained, trap restore.
set -uo pipefail
NL_TAG=lfmcap
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
Q5=/home/witcheer/models/lfm25-24b/LFM2.5-24B-A2B-UD-Q5_K_XL.gguf
nl_check_path "$PY" "$Q5"; nl_gate
v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${v:-99999}" -lt 3000 ] || { nl_say "card busy (${v} MiB), abort"; exit 2; }
restore() { pkill -9 -x llama-server 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; sleep 4; curl -s localhost:8090/health | grep -qi ok && nl_say "Donald proxy OK" || nl_say "Donald proxy NOT ok"; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3
cd "$RIG"; export PYTHONPATH="$RIG"
timeout 1500 $PY -u - "$Q5" <<'PYEOF'
import sys, json, re, time, urllib.request
from pathlib import Path
import lib.quality as q
from lib.evals.gpqa import _build_item, _build_messages
from datasets import load_dataset
path = sys.argv[1]
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
proc = q.start_llama_server(path, ctx_size=24576)
def rep_score(s):
    tail = s[-4000:]
    if len(tail) < 400: return 0.0
    chunks = [tail[i:i+80] for i in range(0, len(tail) - 80, 40)]
    return round(1 - len(set(chunks)) / len(chunks), 3)
out = []
try:
    for idx in [8, 10, 15]:
        msgs = _build_messages(_build_item(ds[idx], idx))
        for mode, extra in [("greedy", {"temperature": 0}), ("liquid", {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05, "seed": 42})]:
            body = {"model": "x", "messages": msgs, "max_tokens": 16384, **extra}
            t0 = time.time()
            r = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            j = json.loads(urllib.request.urlopen(r, timeout=900).read())
            m = j["choices"][0]["message"]; reas = m.get("reasoning_content") or ""; cont = m.get("content") or ""
            rec = {"idx": idx, "mode": mode, "finish": j["choices"][0].get("finish_reason"), "completion_tokens": j["usage"]["completion_tokens"],
                   "repeat_frac_tail": rep_score(reas + cont), "content_tail": cont[-160:], "reasoning_tail": reas[-400:], "s": round(time.time() - t0)}
            out.append(rec)
            print(json.dumps({k: rec[k] for k in ("idx", "mode", "finish", "completion_tokens", "repeat_frac_tail", "s")}), flush=True)
            print("   tail:", repr((reas + cont)[-300:]), flush=True)
finally:
    q.stop_llama_server(proc)
Path("/home/witcheer/lfm-captail.json").write_text(json.dumps(out, indent=2))
PYEOF
nl_say "DONE rc=$?"
