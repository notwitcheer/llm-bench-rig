#!/usr/bin/env python3
"""hermes-settings results -> dataset/hermes_settings*.csv (and a paired-test check against the item's own readout).

Reads the item's result dirs (results/hermes-settings/<leg>/ with summary.json + items.jsonl or multi_turn.jsonl, fit/summary.json,
paired-readout.txt) and the three toolcall-v1 Q6_K baseline legs (results/toolcall-v1/qwen3-8-27b-q6-k-{st-on,st-off,s5-on}).
Writes four csvs (CRLF, like toolcall_v1.csv):

    hermes_settings.csv              one row per leg: setting, score, Wilson 95%, tokens, wall time, paired result against the baseline
    hermes_settings_fit.csv          one row per (KV type, context, request): VRAM after load, prefill and decode tok/s at depth, passkey
    hermes_settings_tasks_s5.csv     one row per multi-turn task: correct (1/0) and completion tokens for every S5 leg
    hermes_settings_tasks_single.csv one row per unique single-turn request: correct (1/0) for the four single-turn legs

Single-turn numbers here use the 899 unique requests (7 nested-schema items sit in both S1 and S2 and are one reply); the 906-item
convention of reports/toolcall-v1.md is in `single_turn_906`. Paired tests are exact two-sided McNemar (binomial on the discordant
items). "identical" is the item's own definition: single-turn, the same generated text; S5, the same per-step completion-token
sequence and the same decoded calls. When paired-readout.txt is present every paired line it holds is re-derived and compared.

    python3 scripts/hermes_settings_table.py --results ~/benchmark-rig/results/hermes-settings \\
        --baseline ~/benchmark-rig/results/toolcall-v1 --out-dir dataset
"""
import csv, json, math, os, re, sys

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
RES = os.path.expanduser(arg("--results", "~/benchmark-rig/results/hermes-settings"))
BASE = os.path.expanduser(arg("--baseline", "~/benchmark-rig/results/toolcall-v1"))
OUT = os.path.expanduser(arg("--out-dir", "/tmp"))
SLUG = "qwen3-8-27b-q6-k"

# leg id, directory, kind (s5 | st), what changed, KV type, ctx, think
LEGS = [
    ("base-s5-on", f"{BASE}/{SLUG}-s5-on", "s5", "baseline (toolcall-v1, 2026-10-24)", "f16", 40960, "on"),
    ("base-st-on", f"{BASE}/{SLUG}-st-on", "st", "baseline (toolcall-v1, 2026-10-24)", "f16", 24576, "on"),
    ("base-st-off", f"{BASE}/{SLUG}-st-off", "st", "baseline, thinking off (toolcall-v1, 2026-10-24)", "f16", 24576, "off"),
    ("s5-on-f16-r2", f"{RES}/s5-on-f16-r2", "s5", "exact repeat of the baseline", "f16", 40960, "on"),
    ("st-on-f16-r2", f"{RES}/st-on-f16-r2", "st", "exact repeat of the baseline", "f16", 24576, "on"),
    ("s5-on-q8", f"{RES}/s5-on-q8", "s5", "KV cache q8_0", "q8_0", 40960, "on"),
    ("st-on-q8", f"{RES}/st-on-q8", "st", "KV cache q8_0", "q8_0", 24576, "on"),
    ("s5-on-q4", f"{RES}/s5-on-q4", "s5", "KV cache q4_0", "q4_0", 40960, "on"),
    ("s5-off-f16", f"{RES}/s5-off-f16", "s5", "thinking off", "f16", 40960, "off"),
    ("s5-on-f16-c32k", f"{RES}/s5-on-f16-c32k", "s5", "context 32,768", "f16", 32768, "on"),
    ("s5-on-f16-c64k", f"{RES}/s5-on-f16-c64k", "s5", "context 65,536", "f16", 65536, "on"),
    ("s5-on-f16-c128k", f"{RES}/s5-on-f16-c128k", "s5", "context 131,072", "f16", 131072, "on"),
]
# (other leg, base leg) pairs, in the order of the item's paired-readout.txt, plus the single-turn thinking pair (not in the readout)
PAIRS = [("s5-on-f16-r2", "base-s5-on"), ("s5-on-q8", "base-s5-on"), ("s5-on-q8", "s5-on-f16-r2"), ("s5-off-f16", "base-s5-on"),
         ("st-on-f16-r2", "base-st-on"), ("st-on-q8", "base-st-on"), ("st-on-q8", "st-on-f16-r2"), ("s5-on-f16-c32k", "base-s5-on"),
         ("s5-on-f16-c64k", "base-s5-on"), ("s5-on-f16-c128k", "base-s5-on"), ("s5-on-q4", "base-s5-on"), ("s5-on-q4", "s5-on-q8"),
         ("base-st-off", "base-st-on")]
READOUT_NAME = {"base-s5-on": f"{SLUG}-s5-on", "base-st-on": f"{SLUG}-st-on", "base-st-off": f"{SLUG}-st-off"}


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


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def ok(r):
    return r.get("outcome") == "correct"


def sig(r):
    return ([s.get("completion_tokens") for s in r.get("steps_log", [])], json.dumps(r.get("decoded"), sort_keys=True))


data = {}  # leg -> dict(rows=ordered list, by_id=dict, summary=dict, kind=...)
for leg, d, kind, *_ in LEGS:
    rows = jl(f"{d}/{'multi_turn' if kind == 's5' else 'items'}.jsonl")
    by_id = {}
    for r in rows:
        by_id[r["id"]] = r  # last wins, as in hermes_settings_paired.py
    data[leg] = dict(rows=rows, by_id=by_id, summary=json.load(open(f"{d}/summary.json")), kind=kind)


def compare(other, base):
    a, b = data[base]["by_id"], data[other]["by_id"]
    kind = data[other]["kind"]
    ids = sorted(set(a) & set(b))
    bo = sum(1 for i in ids if ok(a[i]) and not ok(b[i]))
    oo = sum(1 for i in ids if ok(b[i]) and not ok(a[i]))
    same = (lambda x, y: x.get("generated") == y.get("generated")) if kind == "st" else (lambda x, y: sig(x) == sig(y))
    ident = sum(1 for i in ids if same(a[i], b[i]))
    ct = lambda r: r.get("completion_tokens") or 0
    return dict(paired=len(ids), base_correct=sum(ok(a[i]) for i in ids), other_correct=sum(ok(b[i]) for i in ids),
                base_only_correct=bo, other_only_correct=oo, mcnemar_exact_p=round(binom2(bo, oo), 4), identical_outputs=ident,
                identical_pct=round(100 * ident / len(ids), 1), base_completion_tokens_mean=round(sum(ct(a[i]) for i in ids) / len(ids), 1),
                other_completion_tokens_mean=round(sum(ct(b[i]) for i in ids) / len(ids), 1))


pairs = {(o, b): compare(o, b) for o, b in PAIRS}

# 1. check against the item's own readout
bad = 0
ro_path = f"{RES}/paired-readout.txt"
if os.path.exists(ro_path):
    cur, ro = None, {}
    for ln in open(ro_path):
        ln = ln.strip()
        if ln.startswith("### "):
            cur = ln[4:]
        elif ln.startswith("{") and cur:
            ro[cur] = json.loads(ln)
    for (o, b), r in pairs.items():
        key = f"{o} vs {READOUT_NAME.get(b, b)}"
        if key not in ro:
            continue
        for k in ("paired", "base_correct", "other_correct", "base_only_correct", "other_only_correct", "mcnemar_exact_p",
                  "identical_outputs", "identical_pct", "base_completion_tokens_mean", "other_completion_tokens_mean"):
            if ro[key][k] != r[k]:
                bad += 1
                print(f"MISMATCH {key} {k}: readout {ro[key][k]} vs recomputed {r[k]}", file=sys.stderr)
    print(f"paired lines compared with paired-readout.txt: {len(ro)}, mismatches: {bad}", file=sys.stderr)

# 2. hermes_settings.csv
COLS = ["leg", "slice", "what_changed", "kv_cache", "ctx", "think", "correct", "total", "pct", "wilson95_lo", "wilson95_hi",
        "single_turn_906", "completion_tokens_mean", "request_wall_min", "max_prompt_plus_gen", "force_quit", "capped",
        "resumed_after_hard_stop", "baseline_leg", "baseline_only_correct", "leg_only_correct", "mcnemar_exact_p", "identical_pct", "server_bin"]
BASE_OF = {"s5": "base-s5-on", "st": "base-st-on"}
out_rows = []
for leg, d, kind, what, kv, ctx, think in LEGS:
    D = data[leg]
    u = D["by_id"]
    n = len(u)
    c = sum(ok(r) for r in u.values())
    lo, hi = wilson(c, n)
    wall = sum(r.get("wall_s") or 0 for r in u.values())
    sess = D["summary"].get("wall_s_this_session", wall)
    row = dict(leg=leg, slice="S5 multi_turn_base" if kind == "s5" else "S1..S4 single-turn (899 unique requests)", what_changed=what, kv_cache=kv,
               ctx=ctx, think=think, correct=c, total=n, pct=round(100 * c / n, 1), wilson95_lo=lo, wilson95_hi=hi,
               completion_tokens_mean=round(sum(r.get("completion_tokens") or 0 for r in u.values()) / n, 1), request_wall_min=round(wall / 60, 1),
               resumed_after_hard_stop="yes" if sess < 0.9 * wall else "no", server_bin=D["summary"]["server_bin"])
    if kind == "s5":
        row.update(max_prompt_plus_gen=max(r.get("max_prompt_plus_gen", 0) for r in u.values()),
                   force_quit=sum(1 for r in u.values() if r.get("force_quit")), single_turn_906="", capped="")
    else:
        per_set = {}
        for r in u.values():
            for s in r["sets"]:
                per_set[s] = per_set.get(s, 0) + ok(r)
        row.update(single_turn_906=sum(per_set.values()), max_prompt_plus_gen="", force_quit="",
                   capped=sum(1 for r in u.values() if r.get("truncated")))
    if leg in ("base-s5-on", "base-st-on"):
        row.update(baseline_leg="", baseline_only_correct="", leg_only_correct="", mcnemar_exact_p="", identical_pct="")
    else:
        # the thinking-off baseline row is compared with the thinking-on baseline; every other leg with its own baseline
        b = "base-st-on" if leg == "base-st-off" else BASE_OF[kind]
        r = pairs[(leg, b)]
        row.update(baseline_leg=b, baseline_only_correct=r["base_only_correct"], leg_only_correct=r["other_only_correct"],
                   mcnemar_exact_p=r["mcnemar_exact_p"], identical_pct=r["identical_pct"])
    out_rows.append(row)


def write(name, cols, rows):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/{name} ({len(rows)} rows)", file=sys.stderr)


write("hermes_settings.csv", COLS, out_rows)

# 3. fit csv
fit = json.load(open(f"{RES}/fit/summary.json"))
frows = []
for cell in sorted(fit["cells"], key=lambda c: (c["kv"], c["ctx"])):
    for q in cell["requests"]:
        if "prefill_tok_s" not in q:
            continue
        frows.append(dict(kv_cache=cell["kv"], ctx=cell["ctx"], plan=cell["plan"], build=cell["build_info"], slots=cell["total_slots"],
                          vram_mib_loaded=cell["vram_mib_loaded"], vram_mib_peak_seen=cell["vram_mib_peak_seen"],
                          prompt_tokens=q["prompt_tokens_server"], needle_at=q["needle_frac"], passkey_found=q["found"],
                          prefill_tok_s=q["prefill_tok_s"], decode_tok_s=q["decode_tok_s"]))
write("hermes_settings_fit.csv", list(frows[0].keys()), frows)

# 4. per-task csvs
S5L = ["base-s5-on", "s5-on-f16-r2", "s5-on-q8", "s5-on-q4", "s5-off-f16", "s5-on-f16-c32k", "s5-on-f16-c64k", "s5-on-f16-c128k"]
ids = sorted(data["base-s5-on"]["by_id"], key=lambda s: int(s.rsplit("_", 1)[1]))
trows = []
for i in ids:
    r = dict(id=i, turns_gt=data["base-s5-on"]["by_id"][i]["turns_gt"])
    for l in S5L:
        x = data[l]["by_id"][i]
        r[f"{l}_correct"] = int(ok(x))
        r[f"{l}_completion_tokens"] = x.get("completion_tokens")
    trows.append(r)
write("hermes_settings_tasks_s5.csv", list(trows[0].keys()), trows)
STL = ["base-st-on", "st-on-f16-r2", "st-on-q8", "base-st-off"]
sids = sorted(data["base-st-on"]["by_id"])
srows = []
for i in sids:
    r = dict(id=i, sets="+".join(data["base-st-on"]["by_id"][i]["sets"]))
    for l in STL:
        r[f"{l}_correct"] = int(ok(data[l]["by_id"][i]))
    srows.append(r)
write("hermes_settings_tasks_single.csv", list(srows[0].keys()), srows)

print(json.dumps({f"{o} vs {b}": v for (o, b), v in pairs.items()}, indent=1))
sys.exit(1 if bad else 0)
