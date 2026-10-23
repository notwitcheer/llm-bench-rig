# bolt-gpqa32k.py 2026-10-16: re-run ONLY the GPQA items Bolt capped at 16k, with max_tokens 32768 (ctx 40960),
# same prompts/parser/greedy as the board. writes results/<slug>-thinkon-capped32k/{gpqa_capped32k.json,progress}.
# usage: python bolt-gpqa32k.py <gguf> <src_results_dir> <out_dir>
import sys, json, os, time
from pathlib import Path
import lib.config as _c
_c.load_config()['llama_cpp']['server_bin'] = os.environ['BOLT_SRV']
import lib.quality as q
from lib.evals.base import LLMClient, chat_or_error, parse_choice
from lib.evals.gpqa import _build_item, _build_messages, LETTERS, CAP_SLACK
from datasets import load_dataset
gguf, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
MAXT, CTX = 32768, 40960
prev = json.load(open(src / "gpqa_progress.json"))["completed"]
capped = sorted((int(k) for k, v in prev.items() if v.get("capped")))
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
items = {i: _build_item(ds[i], i) for i in capped}
pp = out / "progress.json"
done = json.load(open(pp)) if pp.exists() else {}
print(f"[g32] capped items {len(capped)}, done {len(done)}", flush=True)
proc = q.start_llama_server(gguf, ctx_size=CTX)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(gguf).stem, think=True, timeout=3600) as client:
        for i in capped:
            k = str(i)
            if k in done: continue
            resp, err = chat_or_error(client, _build_messages(items[i]), max_tokens=MAXT)
            if err: print(f"[g32] {i} error {err}", flush=True); continue
            tok = getattr(client, "last_completion_tokens", None)
            pred = parse_choice(resp); exp = LETTERS[items[i]["answer"]]
            done[k] = {"correct": pred == exp, "predicted": pred, "expected": exp, "completion_tokens": tok,
                       "capped": isinstance(tok, int) and tok >= MAXT - CAP_SLACK, "was_16k": prev[k]}
            json.dump(done, open(pp, "w"))
            print(f"[g32] {i} tok={tok} pred={pred} exp={exp} ok={pred==exp}", flush=True)
finally:
    q.stop_llama_server(proc)
fixed = sum(v["correct"] for v in done.values())
base_correct = sum(1 for v in prev.values() if v["correct"])
summary = {"capped_at_16k": len(capped), "rerun_done": len(done), "rerun_correct": fixed,
           "still_capped_32k": sum(v["capped"] for v in done.values()),
           "board16k_correct": base_correct, "combined_correct": base_correct + fixed, "total": len(prev),
           "combined_score": round(100 * (base_correct + fixed) / len(prev), 2),
           "regime": "think-on greedy; 16k board result for uncapped items + 32k re-run of the 16k-capped items (ctx 40960)",
           "server_bin": "llama.cpp cd0a5b6a1 ~/llama.cpp-bolt"}
json.dump(summary, open(out / "gpqa_capped32k.json", "w"), indent=2); print("[g32] SUMMARY", json.dumps(summary), flush=True)
