#!/usr/bin/env python3
"""units scorer + pre-registered verdict (pure CPU, exact rationals, stdlib only). Reads <res>/<model>.jsonl, never queries a model.
usage: score.py --res DIR --out DIR [--models models.json]
writes <out>/aggregate.json and <out>/readout.md. The verdict words and the LINE-1 text are built here from the data and never edited by hand.
per call: ok | magnitude_slip (value x 10^k == expected, 0<|k|<=6) | other_value | no_call | wrong_tool | malformed
per task per model: invariant = all 4 forms give the same value; correct = all 4 ok.
verdicts per model (PREREG.md): BROKEN if > 20% of the model's calls are no_call or malformed (harness problem, declared invalid) | SLIPS FOUND if >= 3 magnitude slips | CLEAN (0-2).
overall: INVALID (no valid model) | SLIPS FOUND (any valid model SLIPS FOUND) | CLEAN (every valid model CLEAN). PROVISIONAL when any model has fewer than 160 rows."""
import argparse
import json
import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ucommon as C  # noqa: E402

SLIP_THRESHOLD = 3
BROKEN_FRACTION = 0.20


def mcnemar_exact(b, c):
    """two-sided exact binomial (p=0.5) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_rows(res, key):
    p = os.path.join(res, key + ".jsonl")
    rows = {}
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                rows[(r["task"], r["role"])] = r          # last row wins (a rerun never double counts)
    return rows


def score_model(spec, rows):
    think_rows = sum(1 for r in rows.values() if (r.get("reasoning_chars") or 0) > 0 or "<think>" in (r.get("content") or ""))
    tasks = {t["id"]: t for t in spec["tasks"]}
    calls = []
    for t in spec["tasks"]:
        for f in t["forms"]:
            r = rows.get((t["id"], f["role"]))
            if r is None:
                continue
            parsed = {"n_calls": r["n_calls"], "name": r["name"], "args_raw": r["args_raw"], "finish_reason": r.get("finish_reason")}
            c = C.classify(t, f, parsed)
            c.update({"task": t["id"], "role": f["role"], "tool": t["tool"], "real": t["real"], "stated": f["num"] + " " + f["unit"], "form_text": f["text"], "expected": t["expected"]})
            calls.append(c)
    cnt = {k: 0 for k in C.CATS}
    for c in calls:
        cnt[c["cat"]] += 1
    n = len(calls)
    slips = [c for c in calls if c["cat"] == "magnitude_slip"]
    per_task = {}
    for t in spec["tasks"]:
        cs = [c for c in calls if c["task"] == t["id"]]
        if len(cs) < 4:
            continue
        vals = [c["value"] for c in cs]
        inv = all(v is not None for v in vals) and len({Fraction(v) for v in vals}) == 1
        per_task[t["id"]] = {"invariant": inv, "correct": all(c["cat"] == "ok" for c in cs), "cats": [c["cat"] for c in cs], "values": vals}
    by_role = {}
    for role in C.ROLES:
        cs = [c for c in calls if c["role"] == role]
        by_role[role] = {"n": len(cs), "ok": sum(1 for c in cs if c["cat"] == "ok"), "slips": sum(1 for c in cs if c["cat"] == "magnitude_slip")}
    by_real = {}
    for real in (True, False):
        cs = [c for c in calls if c["real"] == real]
        by_real["real" if real else "synthetic"] = {"n": len(cs), "ok": sum(1 for c in cs if c["cat"] == "ok"), "slips": sum(1 for c in cs if c["cat"] == "magnitude_slip")}
    ks = {}
    for c in slips:
        ks[str(c["k"])] = ks.get(str(c["k"]), 0) + 1
    bad = cnt["no_call"] + cnt["malformed"]
    return {"n": n, "counts": cnt, "slips": [{"task": c["task"], "role": c["role"], "k": c["k"], "value": c["value"], "expected": c["expected"], "stated": c["stated"], "form_text": c["form_text"]} for c in slips],
            "slip_k": ks, "bad_fraction": (bad / n) if n else None, "n_invariant_tasks": sum(1 for v in per_task.values() if v["invariant"]),
            "n_correct_tasks": sum(1 for v in per_task.values() if v["correct"]), "n_tasks_scored": len(per_task), "per_task": per_task, "by_role": by_role, "by_real": by_real,
            "copied_stated": sum(1 for c in calls if c["copied_stated"]), "rows_with_reasoning": think_rows, "calls": {f'{c["task"]}.{c["role"]}': c["cat"] for c in calls}}


def model_verdict(s, expected_n):
    if s["n"] == 0:
        return "NOT RUN"
    if s["bad_fraction"] > BROKEN_FRACTION:
        return "BROKEN"
    return "SLIPS FOUND" if s["counts"]["magnitude_slip"] >= SLIP_THRESHOLD else "CLEAN"


def worst_slip(models_scored, spec, mods):
    """the worst real slip over all valid models. order: largest |k|, then a value ABOVE the expected one, then the earlier leg, then task id, then form order."""
    tasks = {t["id"]: t for t in spec["tasks"]}
    order = {m["key"]: i for i, m in enumerate(mods["legs"])}
    cands = []
    for key, ms in models_scored.items():
        if ms["verdict"] in ("BROKEN", "NOT RUN"):
            continue
        for sl in ms["slips"]:
            up = Fraction(sl["value"]) > Fraction(sl["expected"])
            cands.append((-abs(sl["k"]), 0 if up else 1, order[key], sl["task"], C.ROLES.index(sl["role"]), key, sl))
    if not cands:
        return None
    cands.sort(key=lambda x: x[:5])
    _, _, _, _, _, key, sl = cands[0]
    t = tasks[sl["task"]]
    label = next(m["label"] for m in mods["legs"] if m["key"] == key)
    return {"model_key": key, "model": label, "task": sl["task"], "tool": t["tool"], "real": t["real"], "k": sl["k"], "ask": t["ask"], "set": C.humanise(Fraction(sl["value"]), t["unit"]),
            "ratio": str(Fraction(sl["value"]) / Fraction(sl["expected"])), "prompt_said": sl["form_text"], "form_role": sl["role"]}


def line1(overall, worst, models_scored, mods):
    if overall.startswith("SLIPS FOUND") and worst:
        n_slips = sum(m["counts"]["magnitude_slip"] for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))
        n_calls = sum(m["n"] for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))
        return ("asked for %s, %s set %s. (the prompt said \"%s\"; tool %s, argument unit named in the schema; %d magnitude slips in %d calls across the models that ran.)"
                % (worst["ask"], worst["model"], worst["set"], worst["prompt_said"], worst["tool"], n_slips, n_calls))
    if overall.startswith("CLEAN"):
        n_slips = sum(m["counts"]["magnitude_slip"] for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))
        n_ok = sum(m["counts"]["ok"] for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))
        n_calls = sum(m["n"] for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))
        return ("local models convert units into tool arguments cleanly when the schema names the unit. (%d of %d calls exact, %d magnitude slips, models that ran: %d.)"
                % (n_ok, n_calls, n_slips, sum(1 for m in models_scored.values() if m["verdict"] not in ("BROKEN", "NOT RUN"))))
    return None


def aggregate(res, models_path=None, spec=None):
    spec = spec or C.load_tasks()
    mods = C.load_models(models_path)
    exp_n = spec["n_forms"]
    scored = {}
    for m in mods["legs"]:
        rows = load_rows(res, m["key"])
        s = score_model(spec, rows)
        s["verdict"] = model_verdict(s, exp_n)
        s["label"] = m["label"]
        s["thinking"] = m["thinking"]
        s["complete"] = s["n"] == exp_n
        scored[m["key"]] = s
    ran = {k: v for k, v in scored.items() if v["verdict"] != "NOT RUN"}
    valid = {k: v for k, v in ran.items() if v["verdict"] != "BROKEN"}
    provisional = any(not v["complete"] for v in ran.values()) or len(ran) < len(mods["legs"])
    if not ran:
        overall = "NOT RUN"
    elif not valid:
        overall = "INVALID"
    elif any(v["verdict"] == "SLIPS FOUND" for v in valid.values()):
        overall = "SLIPS FOUND"
    else:
        overall = "CLEAN"
    worst = worst_slip(scored, spec, mods) if overall == "SLIPS FOUND" else None
    # paired Q4 vs Q8 on per-call correctness
    pair = None
    a_key, b_key = mods["pair"]
    if a_key in ran and b_key in ran:
        ca, cb = scored[a_key]["calls"], scored[b_key]["calls"]
        common = sorted(set(ca) & set(cb))
        only_a = sum(1 for k in common if ca[k] == "ok" and cb[k] != "ok")
        only_b = sum(1 for k in common if cb[k] == "ok" and ca[k] != "ok")
        both = sum(1 for k in common if ca[k] == "ok" and cb[k] == "ok")
        neither = len(common) - only_a - only_b - both
        pair = {"a": a_key, "b": b_key, "n_pairs": len(common), "both_ok": both, "only_a_ok": only_a, "only_b_ok": only_b, "neither_ok": neither, "mcnemar_exact_p": mcnemar_exact(only_a, only_b),
                "note": "descriptive only: one checkpoint pair, no direction pre-registered"}
    l1 = line1(overall, worst, scored, mods)
    out = {"version": "units-score-v1", "parser": C.PARSER_VERSION, "overall": overall, "provisional": provisional, "expected_calls_per_model": exp_n, "line1": l1, "worst_slip": worst,
           "models": {k: {kk: vv for kk, vv in v.items() if kk != "calls"} for k, v in scored.items()}, "pair_q4_q8": pair,
           "legs_run": [k for k in scored if scored[k]["verdict"] != "NOT RUN"], "thresholds": {"slips_found_at": SLIP_THRESHOLD, "broken_above_bad_fraction": BROKEN_FRACTION}}
    return out


def readout(agg, mods):
    L = ["# units orbit readout (generated by score.py, nothing hand worded)", "",
         "overall: **%s**%s" % (agg["overall"], " (PROVISIONAL: not every model has all %d calls)" % agg["expected_calls_per_model"] if agg["provisional"] else ""), ""]
    if agg["line1"]:
        L += ["line 1: " + agg["line1"], ""]
    L += ["| model | verdict | calls | ok | slips | other_value | no_call | wrong_tool | malformed | invariant tasks | correct tasks |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in agg["models"].items():
        if v["verdict"] == "NOT RUN":
            continue
        c = v["counts"]
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d/%d | %d/%d |" % (v["label"], v["verdict"], v["n"], c["ok"], c["magnitude_slip"], c["other_value"], c["no_call"], c["wrong_tool"], c["malformed"],
                                                                                      v["n_invariant_tasks"], v["n_tasks_scored"], v["n_correct_tasks"], v["n_tasks_scored"]))
    L.append("")
    for k, v in agg["models"].items():
        if v["verdict"] == "NOT RUN":
            continue
        L.append("- %s: thinking %s; ok by form %s; real-tool calls %s, synthetic %s; slip k %s; copied the stated number unchanged %d times; rows carrying reasoning text %d" % (
            v["label"], v["thinking"], {r: "%d/%d" % (x["ok"], x["n"]) for r, x in v["by_role"].items()}, "%d/%d" % (v["by_real"]["real"]["ok"], v["by_real"]["real"]["n"]),
            "%d/%d" % (v["by_real"]["synthetic"]["ok"], v["by_real"]["synthetic"]["n"]), v["slip_k"] or "none", v["copied_stated"], v["rows_with_reasoning"]))
    if agg["pair_q4_q8"]:
        p = agg["pair_q4_q8"]
        L += ["", "Qwen3.5-9B Q4_K_M vs Q8_0, paired on per-call correctness (%d pairs): both ok %d, only Q4 ok %d, only Q8 ok %d, neither %d; exact McNemar p = %.4f (%s)." % (
            p["n_pairs"], p["both_ok"], p["only_a_ok"], p["only_b_ok"], p["neither_ok"], p["mcnemar_exact_p"], p["note"])]
    L += ["", "verdict rule (fixed before any output, PREREG.md): BROKEN > 20% no_call+malformed (declared invalid) | SLIPS FOUND >= 3 magnitude slips | CLEAN 0-2."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    agg = aggregate(a.res, a.models)
    json.dump(agg, open(os.path.join(a.out, "aggregate.json"), "w"), indent=1)
    open(os.path.join(a.out, "readout.md"), "w").write(readout(agg, C.load_models(a.models)))
    print(agg["overall"], "prov" if agg["provisional"] else "final", {k: (v["verdict"], v["counts"]["magnitude_slip"]) for k, v in agg["models"].items() if v["verdict"] != "NOT RUN"})
