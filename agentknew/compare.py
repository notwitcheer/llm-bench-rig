import json, os, numpy as np, pandas as pd
W = os.path.expanduser("~/esoteric/nerve-gate")
P = json.load(open(f"{W}/paper/summary.json")); O = json.load(open(f"{W}/ours_v57_summary.json"))
PF = json.load(open(f"{W}/paper/gate_fit_gptoss20b.json")); OF = json.load(open(f"{W}/ours_gate_fit.json"))
ps = pd.read_parquet(f"{W}/paper/scores_v57.parquet"); ps = ps[ps.model == "gptoss20b"]
os_ = pd.read_parquet(f"{W}/ours_scores_v57.parquet")
k = ["set", "episode_id", "event_id"]; m = ps.merge(os_, on=k, suffixes=("_paper", "_ours"))
from scipy.stats import spearmanr
r = {"dev_fit": {"paper": PF, "ours": OF},
     "conf_per_model_gptoss20b": {"paper": P["per_model"].get("gptoss20b"), "ours": O["per_model"].get("gptoss20b")},
     "conf_n": {"paper": ps.cls.value_counts().to_dict(), "ours": O["n"]},
     "fire_rate": {"paper": P["gate_fire_rate"].get("gptoss20b"), "ours": O["gate_fire_rate"].get("gptoss20b")},
     "ours_obfuscate_gptoss": O.get("obfuscate"),
     "paper_obfuscate_gptoss_only": None,
     "unit_overlap": int(len(m)),
     "spearman_I_ours_vs_paper": float(spearmanr(m.I_paper, m.I_ours).correlation) if len(m) > 10 else None,
     "E1": O.get("E1")}
try:
    import importlib.util, sys
    sys.path.insert(0, os.path.expanduser("~/esoteric/Honesty/src"))
    s = importlib.util.spec_from_file_location("m150", os.path.expanduser("~/esoteric/Honesty/scripts/150_v55_analysis.py")); M = importlib.util.module_from_spec(s); s.loader.exec_module(M)
    ob = ps[ps.variant == "obfuscate"]; r["paper_obfuscate_gptoss_only"] = {c: round(M.auc(ob, c), 4) for c in ("I", "T_sup", "M_cot") if c in ob}
except Exception as e:
    r["paper_obfuscate_gptoss_only"] = repr(e)[:200]
json.dump(r, open(f"{W}/compare.json", "w"), indent=1, default=str); print(json.dumps(r, indent=1, default=str))
