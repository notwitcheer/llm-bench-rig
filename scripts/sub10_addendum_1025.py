#!/usr/bin/env python3
"""sub-10 GB Qwen3.8-27B study, 2026-10-25 addendum: GPQA Diamond (think-on 16k) + AIME 2026 / HMMT Feb 2026 (think-on 32k)
for Underdog-Saluki-27B IQ2-mix and Qwen3.8-27B GSQ-RCO IQ2_XS, against the banked Q6_K and UD-IQ3_XXS rows.

Runs on the rig, reads only per-item progress files + detail jsons (never the GPU). Prints json: per-cell counts (correct, capped,
capped-wrong, finished-right), exact two-sided binomial paired tests on the discordant items, and the matharena csv rows for the
two new slugs (column order of dataset/matharena_contest_32k.csv).

    python3 scripts/sub10_addendum_1025.py [--results ~/benchmark-rig/results]
"""
import json, math, os, sys

R = os.path.expanduser(sys.argv[sys.argv.index("--results") + 1] if "--results" in sys.argv else "~/benchmark-rig/results")
ROWS = {  # key -> (slug, gpqa dir, capped source)
    "saluki": "underdog-saluki-27b-iq2-mix",
    "gsq": "qwen3-8-27b-gsq-rco-iq2-xs",
    "q6": "qwen3-8-27b-q6-k",
    "iq3": "qwen3-8-27b-ud-iq3-xxs",
}
GPQA_DIR = {"saluki": "underdog-saluki-27b-iq2-mix-thinkon", "gsq": "qwen3-8-27b-gsq-rco-iq2-xs-thinkon",
            "q6": "qwen3-8-27b-q6-k-thinkon-rerun", "iq3": "qwen3-8-27b-ud-iq3-xxs-thinkon"}
CAP16 = 16384 - 8  # legacy legs without the per-item `capped` flag: completion >= 16,376 tokens in the sidecar
TASKS = ["ma_aime_2026", "ma_hmmt_feb_2026"]
MATH500_16K = {"saluki": "underdog-saluki-27b-iq2-mix-thinkon", "gsq": "qwen3-8-27b-gsq-rco-iq2-xs-thinkon"}


def binom2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)


def pair(a, b):
    keys = sorted(set(a) & set(b))
    x = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    y = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    return {"n": len(keys), "a_only": x, "b_only": y, "p": round(binom2(x, y), 4)}


def gpqa_items(key):
    d = f"{R}/{GPQA_DIR[key]}"
    comp = json.load(open(f"{d}/gpqa_progress.json"))["completed"]
    tok = {}
    tp = f"{d}/gpqa_tokens.jsonl"
    if os.path.exists(tp):
        for i, l in enumerate(open(tp)):
            if l.strip():
                j = json.loads(l)
                tok[j.get("idx", i)] = j
    out = {}
    for i, v in comp.items():
        i = int(i)
        cap = v.get("capped")
        if cap is None and i in tok:
            cap = tok[i]["completion_tokens"] >= CAP16
        out[i] = {"correct": bool(v["correct"]), "capped": cap}
    return out


def split(items):
    n = len(items)
    cor = sum(1 for x in items.values() if x["correct"])
    if any(x["capped"] is None for x in items.values()):
        return {"n": n, "correct": cor, "capped": None}
    cap = sum(1 for x in items.values() if x["capped"])
    capc = sum(1 for x in items.values() if x["capped"] and x["correct"])
    finc = sum(1 for x in items.values() if not x["capped"] and x["correct"])
    return {"n": n, "correct": cor, "wilson95": wilson(cor, n), "capped": cap, "capped_rate": round(cap / n, 4),
            "capped_correct": capc, "capped_wrong": cap - capc, "finished": n - cap, "finished_correct": finc}


out = {"gpqa": {}, "gpqa_paired": {}, "contests": {}, "contests_paired": {}, "csv_rows": []}
G = {k: gpqa_items(k) for k in ROWS}
for k in ROWS:
    out["gpqa"][k] = split(G[k])
for a, b in [("gsq", "saluki"), ("q6", "saluki"), ("iq3", "saluki"), ("q6", "gsq"), ("iq3", "gsq")]:
    out["gpqa_paired"][f"{a}_vs_{b}"] = pair(G[a], G[b])  # a_only = first row right, second wrong

M = {}
for k, slug in ROWS.items():
    M[k] = {}
    for t in TASKS:
        d = f"{R}/{slug}-matharena-32k"
        comp = json.load(open(f"{d}/{t}_progress.json"))["completed"]
        M[k][t] = {int(i): {"correct": bool(v["correct"]), "capped": bool(v.get("capped")),
                            "error": v.get("error_request"), "vetoed": bool(v.get("exact_vetoed"))} for i, v in comp.items()}
        det = json.load(open(f"{d}/{t}_detail.json"))
        s = split(M[k][t])
        s.update(score=det["score"], parse_failures=det["parse_failures"], request_errors=det.get("request_errors", 0),
                 exact_vetoes=det.get("exact_vetoes", 0), tokens_per_correct=det["tokens_per_correct"],
                 median_completion=det["completion_tokens_median"],
                 request_error_items=[i for i, x in M[k][t].items() if x["error"]])
        out["contests"].setdefault(k, {})[t] = s
        if k in MATH500_16K:
            m5 = json.load(open(f"{R}/{MATH500_16K[k]}/math500_detail.json"))
            ctx = "think-on, max_tokens 32768, ctx 40960, zero-shot, greedy"
            size = os.stat(os.path.expanduser(
                {"saluki": "~/models/saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf",
                 "gsq": "~/models/gsq-rco-27b/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf"}[k])).st_size
            model, quant = (("Underdog-Saluki-27B-1.0", "IQ2-mix") if k == "saluki" else ("Qwen3.8-27B", "GSQ-RCO IQ2_XS"))
            out["csv_rows"].append([slug, model, quant, size, t.replace("ma_", ""), "acc", det["score"], det["correct"], det["total"],
                                    det["capped_count"], det["capped_rate"], s["capped_correct"], s["finished"], s["finished_correct"],
                                    det["parse_failures"], det.get("exact_vetoes", 0), det["tokens_per_correct"],
                                    det["completion_tokens_median"], m5["score"], "16k", m5["capped_rate"], ctx,
                                    "matharena-contest-32k (sub-10 GB study rows added 2026-10-25, mainline llama.cpp b10371 5d16e81dd)"])
for t in TASKS:
    for a, b in [("gsq", "saluki"), ("q6", "saluki"), ("iq3", "saluki"), ("q6", "gsq"), ("iq3", "gsq")]:
        out["contests_paired"][f"{t}:{a}_vs_{b}"] = pair(M[a][t], M[b][t])
# both contests pooled (63 problems), the way the contest report pairs
for a, b in [("gsq", "saluki"), ("q6", "saluki"), ("iq3", "saluki"), ("q6", "gsq"), ("iq3", "gsq")]:
    x = {(t, i): v for t in TASKS for i, v in M[a][t].items()}
    y = {(t, i): v for t in TASKS for i, v in M[b][t].items()}
    out["contests_paired"][f"both:{a}_vs_{b}"] = pair(x, y)
print(json.dumps(out, indent=1))
