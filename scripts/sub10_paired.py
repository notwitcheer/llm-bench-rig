#!/usr/bin/env python3
"""paired per-item comparison for the sub-10 GB study: Saluki vs the Qwen3.8-27B / Bonsai rows on identical items."""
import json, math, collections
R = "/home/witcheer/benchmark-rig/results"
SUB = "underdog-saluki-27b-iq2-mix"
REFS = ["qwen3-8-27b-q6-k", "qwen3-8-27b-ud-iq3-xxs", "ternary-bonsai-2-27b-pq2-0", "ternary-bonsai-2-27b-ptq1-0"]
TASKS = ["ifeval", "math500", "humaneval_plus", "mbpp_plus"]

def load(slug, t):
    return json.load(open(f"{R}/{slug}-thinkon/{t}_progress.json"))["completed"]

def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

out = {"paired": {}, "ifeval_cats": {}, "capped_wrong": {}}
s = {t: load(SUB, t) for t in TASKS}
for ref in REFS:
    out["paired"][ref] = {}
    tot_b = tot_c = 0
    for t in TASKS:
        r = load(ref, t)
        keys = sorted(set(s[t]) & set(r))
        b = sum(1 for k in keys if r[k]["correct"] and not s[t][k]["correct"])  # ref right, saluki wrong
        c = sum(1 for k in keys if s[t][k]["correct"] and not r[k]["correct"])  # saluki right, ref wrong
        tot_b += b; tot_c += c
        out["paired"][ref][t] = {"n": len(keys), "ref_only": b, "sub_only": c, "p": round(binom_two_sided(b, c), 5)}
    out["paired"][ref]["all"] = {"ref_only": tot_b, "sub_only": tot_c, "p": round(binom_two_sided(tot_b, tot_c), 6)}

# IFEval per-instruction-group pass rates, Saluki vs Q6_K
for slug in [SUB, "qwen3-8-27b-q6-k"]:
    agg = collections.defaultdict(lambda: [0, 0])
    for k, v in load(slug, "ifeval").items():
        for iid, ok in zip(v.get("instruction_ids", []), v.get("per_instruction", [])):
            if ok is None:
                continue
            g = iid.split(":")[0]
            agg[g][0] += int(bool(ok)); agg[g][1] += 1
    out["ifeval_cats"][slug] = {g: v for g, v in sorted(agg.items())}

for t in TASKS:
    wrong = [v for v in s[t].values() if not v["correct"]]
    out["capped_wrong"][t] = {"wrong": len(wrong), "wrong_capped": sum(1 for v in wrong if v.get("capped")),
                              "right_capped": sum(1 for v in s[t].values() if v["correct"] and v.get("capped"))}
print(json.dumps(out, indent=1))
