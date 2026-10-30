"""PROVISIONAL, CPU: 2-fold scenario cross-fit of the paper's internal recipe (scripts/150 fit_internal) on our fresh conf
units only (155 units(), gpt-oss). Not the preregistered frozen-dev protocol; an early signal check."""
import os, sys, importlib.util, json, numpy as np, pandas as pd
R = os.path.expanduser("~/esoteric/Honesty"); os.chdir(R); sys.path.insert(0, R + "/src"); sys.argv = [sys.argv[0]]
s = importlib.util.spec_from_file_location("m155", R + "/scripts/155_v57_analysis.py"); M = importlib.util.module_from_spec(s); s.loader.exec_module(M)
C, X = M.units("gptoss20b", "conf"); k = C.cls.isin(["VIOL", "XOTHER", "XPLUS"]).values; C, X = C[k].reset_index(drop=True), X[k]
C["template"] = C.template + "|" + C.tool
oof = np.full(len(C), np.nan)
for f in (0, 1):
    tr = C.fold.values != f; oof[~tr] = M.M150.score_internal(M.M150.fit_internal(X[tr], C.y.values[tr]), X[~tr])
C["I"] = oof
vp, vo = C[C.cls.isin(["VIOL", "XPLUS"])], C[C.cls.isin(["VIOL", "XOTHER"])]
r = {"n": C.cls.value_counts().to_dict(), "xfit_conf_auroc": {"all": round(M.M150.auc(C, "I"), 4), "vs_XPLUS": round(M.M150.auc(vp, "I"), 4), "vs_XOTHER": round(M.M150.auc(vo, "I"), 4)}}
P = pd.read_parquet(os.path.expanduser("~/esoteric/nerve-gate/paper/scores_v57.parquet")); P = P[P.model == "gptoss20b"]
r["paper_conf_n"] = P.cls.value_counts().to_dict()
m = C.merge(P[["set", "episode_id", "event_id", "I", "T_sup", "M_cot"]], on=["set", "episode_id", "event_id"], suffixes=("", "_paper"))
r["overlap_units"] = len(m)
if len(m):
    from scipy.stats import spearmanr
    r["spearman_ours_xfit_vs_paper_frozen"] = round(float(spearmanr(m.I, m.I_paper).correlation), 4)
    vpm, vom = m[m.cls.isin(["VIOL", "XPLUS"])], m[m.cls.isin(["VIOL", "XOTHER"])]
    r["same_units"] = {lab: {c: round(M.M150.auc(d, c), 4) for c in ("I", "I_paper", "T_sup", "M_cot")} for lab, d in (("all", m), ("vs_XPLUS", vpm), ("vs_XOTHER", vom))}
    ob = m[m.variant == "obfuscate"]; r["obfuscate_same_units"] = {c: round(M.M150.auc(ob, c), 4) for c in ("I", "I_paper", "T_sup", "M_cot")}
print(json.dumps(r, indent=1)); json.dump(r, open(os.path.expanduser("~/esoteric/nerve-gate/early_xfit.json"), "w"), indent=1)
