#!/usr/bin/env python3
"""toolcall-v1 results -> dataset/toolcall_v1.csv + paired tests (json on stdout).

Runs on the rig (needs the six GGUFs for `stat -L` byte sizes and results/toolcall-v1/<slug>-{st-on,st-off,s5-on}/).
One csv row per (slug, leg). Single-turn legs fill S1..S4 + totals; the multi-turn leg (S5) fills s5_correct/s5_total/force_quit.
Single-turn scores follow the board convention of the status notes: S1 439 + S2 67 + S3 300 + S4 100 = 906 scored items. The 899 unique
requests are fewer because 7 nested-schema items sit in both S1 and S2 (one reply, graded under both sets); `unique_correct` /
`unique_requests` carry the de-duplicated count.

    python3 scripts/toolcall_v1_table.py [--results ~/benchmark-rig/results/toolcall-v1] [--out dataset/toolcall_v1.csv]
"""
import csv, json, math, os, sys

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
RES = os.path.expanduser(arg("--results", "~/benchmark-rig/results/toolcall-v1"))
OUT = arg("--out", "/tmp/toolcall_v1.csv")
M = os.path.expanduser("~/models")
ROWS = [  # slug, model, quant, gguf path (the queue item's ROWS= block)
    ("qwen3-8-27b-q6-k", "Qwen3.8-27B", "Q6_K", f"{M}/qwen38-27b/Qwen3.8-27B-Q6_K.gguf"),
    ("qwen3-8-27b-ud-iq3-xxs", "Qwen3.8-27B", "UD-IQ3_XXS", f"{M}/qwen38-27b/Qwen3.8-27B-UD-IQ3_XXS.gguf"),
    ("underdog-saluki-27b-iq2-mix", "Underdog-Saluki-27B-1.0", "IQ2-mix", f"{M}/saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"),
    ("qwen3-8-27b-gsq-rco-iq2-xs", "Qwen3.8-27B", "GSQ-RCO IQ2_XS", f"{M}/gsq-rco-27b/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf"),
    ("ternary-bonsai-2-27b-pq2-0", "Ternary-Bonsai-2-27B", "PQ2_0", f"{M}/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
    ("ternary-bonsai-2-27b-ptq1-0", "Ternary-Bonsai-2-27B", "PTQ1_0", f"{M}/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
]
LEGS = ["st-on", "st-off", "s5-on"]
COLS = ["slug", "model", "quant", "file_bytes", "leg", "s1_correct", "s2_correct", "s3_correct", "s4_correct", "single_turn_correct",
        "single_turn_total", "single_turn_pct", "unique_correct", "unique_requests", "s5_correct", "s5_total", "force_quit",
        "python_bool_replies", "capped", "server_bin", "regime"]
SET_N = {"S1": 439, "S2": 67, "S3": 300, "S4": 100}


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
    return [round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)]


def load_items(slug, leg):
    p = f"{RES}/{slug}-{leg}/items.jsonl"
    return {j["id"]: j for j in (json.loads(l) for l in open(p) if l.strip())}


rows, S5, ST = [], {}, {}
for slug, model, quant, path in ROWS:
    size = os.stat(path).st_size  # stat -L semantics (follows symlinks)
    for leg in LEGS:
        j = json.load(open(f"{RES}/{slug}-{leg}/summary.json"))
        r = dict(slug=slug, model=model, quant=quant, file_bytes=size, leg=leg, server_bin=j["server_bin"], regime=j["regime"])
        if leg == "s5-on":
            mt = j["multi_turn"]
            assert len(mt) == 100, (slug, len(mt))
            S5[slug] = {x["id"]: x["outcome"] == "correct" for x in mt}
            r.update(s5_correct=sum(S5[slug].values()), s5_total=len(mt), force_quit=sum(1 for x in mt if x.get("force_quit")))
        else:
            st = j["single_turn"]
            c = {s: st[s].get("correct", 0) for s in SET_N}
            for s, n in SET_N.items():
                assert sum(st[s].values()) == n, (slug, leg, s)
            tot = sum(c.values())
            it = load_items(slug, leg)
            ST[(slug, leg)] = {k: v["outcome"] == "correct" for k, v in it.items()}
            r.update(s1_correct=c["S1"], s2_correct=c["S2"], s3_correct=c["S3"], s4_correct=c["S4"], single_turn_correct=tot,
                     single_turn_total=sum(SET_N.values()), single_turn_pct=round(100 * tot / sum(SET_N.values()), 1),
                     unique_correct=sum(ST[(slug, leg)].values()), unique_requests=len(it),
                     python_bool_replies=j["single_turn_pace"]["python_bool_replies"], capped=j["single_turn_pace"]["capped"])
        rows.append(r)
assert len(rows) == 18
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in COLS})

q6 = ROWS[0][0]
out = {"s5_wilson95": {s: wilson(sum(S5[s].values()), 100) for s in S5}, "s5_vs_q6k": {}, "s5_other": {}, "st_vs_q6k": {"st-on": {}, "st-off": {}},
       "gsq_on_vs_off": {}}


def pair(a, b):  # a_only / b_only, exact two-sided
    keys = sorted(set(a) & set(b))
    x = sum(1 for k in keys if a[k] and not b[k])
    y = sum(1 for k in keys if b[k] and not a[k])
    return {"n": len(keys), "first_only": x, "second_only": y, "p": round(binom2(x, y), 4)}


for slug, *_ in ROWS[1:]:
    out["s5_vs_q6k"][slug] = pair(S5[q6], S5[slug])  # first = Q6_K right only
    for leg in ("st-on", "st-off"):
        out["st_vs_q6k"][leg][slug] = pair(ST[(q6, leg)], ST[(slug, leg)])
for a, b in [("underdog-saluki-27b-iq2-mix", "qwen3-8-27b-gsq-rco-iq2-xs"), ("qwen3-8-27b-ud-iq3-xxs", "qwen3-8-27b-gsq-rco-iq2-xs"),
             ("ternary-bonsai-2-27b-pq2-0", "underdog-saluki-27b-iq2-mix")]:
    out["s5_other"][f"{a}_vs_{b}"] = pair(S5[a], S5[b])
g = "qwen3-8-27b-gsq-rco-iq2-xs"
out["gsq_on_vs_off"] = pair(ST[(g, "st-on")], ST[(g, "st-off")])  # first = think-on right only
print(json.dumps(out, indent=1))
