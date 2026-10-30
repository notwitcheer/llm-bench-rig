"""CPU: rebuild outputs/rollouts/<set>.parquet for gpt-oss sets the GitHub archives omit (>5 MB) from the HF CAPE-Bench
parquet (events_json -> events), then sample rendered token lengths with the paper's own renderer (scripts/56)."""
import json, os, sys, glob, importlib.util, numpy as np, pandas as pd
R = os.path.expanduser("~/esoteric/Honesty"); sys.path.insert(0, R + "/src"); os.chdir(R)
O = R + "/outputs"; H = os.path.expanduser("~/esoteric/cape-bench/episodes")
made = []
for f in sorted(glob.glob(f"{H}/xm_gptoss20b_*.parquet")):
    st = os.path.basename(f)[:-8]; dst = f"{O}/rollouts/{st}.parquet"
    if os.path.exists(dst):
        continue
    d = pd.read_parquet(f)
    d["events"] = d.events_json.map(json.loads)
    d = d.drop(columns=["events_json"])
    d.to_parquet(dst); made.append((st, len(d)))
print("rebuilt rollouts:", made)
s = importlib.util.spec_from_file_location("x56", R + "/scripts/56_xm_replay_extract.py"); X = importlib.util.module_from_spec(s); s.loader.exec_module(X)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/gpt-oss-20b"))
a = X.ACT["gptoss20b"]
for bench, st in [("core24b", "xm_gptoss20b_core24b_minus6789"), ("core24b", "xm_gptoss20b_core24b_plus"), ("core24", "xm_gptoss20b_core24_minus")]:
    specs = {}
    for l in open(f"{O}/capebench/{bench}/scenarios.jsonl"):
        x = json.loads(l); specs[x["instance_id"]] = x
    E = pd.read_parquet(f"{O}/rollouts/{st}.parquet").sample(150, random_state=0)
    n = []
    for ep in E.itertuples():
        ev = list(ep.events); acts = X.action_events(ep)
        if not acts: continue
        full = X.render(tok, X.messages_upto(specs[ep.instance_id], ev, acts[-1]["event_id"], a["suffix"]), a["kwargs"])
        n.append(len(tok(full, add_special_tokens=False).input_ids))
    n = np.array(n); print(st, "sampled", len(n), "tokens q50/q90/q99/max", np.percentile(n, [50, 90, 99]).round().tolist(), n.max(), ">6k", (n > 6000).mean().round(3), ">8k", (n > 8000).mean().round(3))
