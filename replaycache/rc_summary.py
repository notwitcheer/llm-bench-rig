#!/usr/bin/env python3
"""rc_summary.py <run_dir> <sel.json> [--mock]  |  rc_summary.py --selftest
Applies the PRE-REGISTERED rule (PREREG.md) to a finished run. Nothing is worded by hand: the verdict words and line 1 come out of decide()/line1() below.
Writes <run>/summary.json (the artefact the item skips on) + <run>/out/summary.txt. Prints VERDICT and LINE1 lines for the item log."""
import json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_common import wilson, mcnemar_exact, pct

HIT_FLOOR = 0.90         # threshold: the tightest s that keeps >= 90% of held-out main as served hits
HOLD_A_PTS = 3.0         # CACHE HOLDS: B(gated) >= A - 3 pts on held-out main ...
HOLD_C_PTS = 20.0        # ... AND B(gated) >= C + 20 pts
TRAP_RATE = 0.20         # NEAR-MISS TRAP: >= 20% of p1 queries are cache hits that return a wrong answer
MIN_ENTRIES = 10
MIN_A_ACC = 0.50


def decide(st):
    """st: counts only. returns (verdict, holds, trap). order matters: a finished build with too few entries is BROKEN without needing the arms."""
    n, nn = st["n_held"], st["n_near"]
    if st.get("build_complete") and st["entries"] < st.get("min_entries", MIN_ENTRIES):
        return "BROKEN", None, None
    if st["incomplete"]:
        return "INCOMPLETE", None, None
    if st["A_held_k"] < MIN_A_ACC * n:
        return "BROKEN", None, None
    a, b, c = 100.0 * st["A_held_k"] / n, 100.0 * st["Bg_held_k"] / n, 100.0 * st["Crule_held_k"] / n
    holds = (b >= a - HOLD_A_PTS) and (b >= c + HOLD_C_PTS)
    trap = st["near_hit_wrong_k"] >= TRAP_RATE * nn
    v = {(True, True): "HOLDS AND TRAPS", (True, False): "HOLDS, NO TRAP", (False, True): "DOES NOT HOLD, TRAPS", (False, False): "DOES NOT HOLD, NO TRAP"}[(holds, trap)]
    return v, holds, trap


def line1(v, st):
    n, nn = st["n_held"], st["n_near"]
    A, B, C = pct(st["A_held_k"] / n), pct(st["Bg_held_k"] / n), pct(st["Crule_held_k"] / n)
    X = pct(st["near_hit_wrong_k"] / nn)
    head = "NVIDIA says a CPU can replay a big model's reasoning from a cache. I rebuilt it on my 5090"
    if v == "HOLDS AND TRAPS":
        return "%s: on new questions it scores %s%% against %s%% for the 27B reasoning from scratch, but on near-miss questions it was confidently wrong %s%% of the time." % (head, B, A, X)
    if v == "HOLDS, NO TRAP":
        return "%s and it holds: %s%% against %s%% for the 27B from scratch on new questions, and on near-miss questions it was confidently wrong only %s%% of the time." % (head, B, A, X)
    if v == "DOES NOT HOLD, TRAPS":
        return "%s and it does not hold here: %s%% against %s%% for the 27B from scratch (a stock 1B alone gets %s%%), and on near-miss questions it was confidently wrong %s%% of the time." % (head, B, A, C, X)
    if v == "DOES NOT HOLD, NO TRAP":
        return "%s and it does not hold here: %s%% against %s%% for the 27B from scratch (a stock 1B alone gets %s%%). near-miss questions did not fool it: confidently wrong %s%% of the time." % (head, B, A, C, X)
    return ""


def load_dir(d, ext=".json"):
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(ext) and not f.startswith("."):
                out[f[:-len(ext)]] = json.load(open(os.path.join(d, f)))
    return out


def compute(run, sel):
    A = load_dir(os.path.join(run, "results", "A"))
    B = load_dir(os.path.join(run, "cpu", "B"))
    C = load_dir(os.path.join(run, "cpu", "C"))
    ents = {}
    ed = os.path.join(run, "entries")
    nfail = 0
    if os.path.isdir(ed):
        for f in os.listdir(ed):
            if f.endswith(".failed.json"):
                nfail += 1
            elif f.endswith(".json"):
                ents[int(f[:-5])] = json.load(open(os.path.join(ed, f)))
    keys = {"held": [], "near": []}
    for t in sel["templates"]:
        for kind in ("held", "near"):
            for sp, tid, inst in t[kind]:
                keys[kind].append("%s-%d-%02d" % (kind, tid, inst))
    n, nn = len(keys["held"]), len(keys["near"])
    incomplete = any(k not in A or k not in B or k not in C for k in keys["held"] + keys["near"])
    build_complete = (len(ents) + nfail) >= len(sel["templates"])
    have = lambda d, ks: [k for k in ks if k in d]
    # ---- threshold from held-out main only
    served_s = sorted([B[k]["s"] for k in keys["held"] if k in B and B[k]["served"]], reverse=True)
    need = math.ceil(HIT_FLOOR * n)
    unreachable = len(served_s) < need
    tau = (served_s[-1] if served_s else 1.0) if unreachable else served_s[need - 1]
    hit = lambda k: (k in B) and B[k]["served"] and B[k]["s"] >= tau - 1e-12

    def arm(kind):
        ks = keys[kind]
        r = {"n": len(ks)}
        r["A_k"] = sum(1 for k in ks if k in A and A[k]["correct"])
        r["hit_k"] = sum(1 for k in ks if hit(k))
        r["hit_right_k"] = sum(1 for k in ks if hit(k) and B[k]["correct"])
        r["hit_wrong_k"] = r["hit_k"] - r["hit_right_k"]
        r["Bg_k"] = sum(1 for k in ks if (B[k]["correct"] if hit(k) else (k in A and A[k]["correct"])))
        r["Bc_k"] = r["hit_right_k"]
        r["C_k"] = sum(1 for k in ks if k in C and C[k]["C"]["correct"])
        r["Cd_k"] = sum(1 for k in ks if k in C and C[k]["Cd"]["correct"])
        r["Crule_k"] = max(r["C_k"], r["Cd_k"])
        r["b_AonlyRight"] = sum(1 for k in ks if k in A and A[k]["correct"] and not (B[k]["correct"] if hit(k) else A[k]["correct"]))
        r["c_BonlyRight"] = sum(1 for k in ks if k in A and (not A[k]["correct"]) and (B[k]["correct"] if hit(k) else A[k]["correct"]))
        r["b_AonlyRight_cacheonly"] = sum(1 for k in ks if k in A and A[k]["correct"] and not (hit(k) and B[k]["correct"]))
        r["c_BonlyRight_cacheonly"] = sum(1 for k in ks if k in A and (not A[k]["correct"]) and (hit(k) and B[k]["correct"]))
        r["mcnemar_gated_p"] = mcnemar_exact(r["b_AonlyRight"], r["c_BonlyRight"])
        r["mcnemar_cacheonly_p"] = mcnemar_exact(r["b_AonlyRight_cacheonly"], r["c_BonlyRight_cacheonly"])
        for nm in ("A", "Bg", "Bc", "C", "Cd", "hit", "hit_wrong"):
            kk = r[nm + "_k"]; lo, hi = wilson(kk, len(ks)); r[nm + "_acc"] = kk / max(1, len(ks)); r[nm + "_ci"] = [lo, hi]
        sv = [B[k]["s"] for k in ks if k in B]
        r["s_median"] = statistics.median(sv) if sv else None
        r["routing_to_own_entry"] = sum(1 for k in ks if k in B and B[k]["entry"] == B[k]["tid"]) / max(1, len(ks))
        r["cpu_ms_B_median"] = statistics.median([B[k]["cpu_ms"] for k in ks if k in B]) if any(k in B for k in ks) else None
        r["cpu_ms_B_mean"] = statistics.mean([B[k]["cpu_ms"] for k in ks if k in B]) if any(k in B for k in ks) else None
        r["lm_tokens_B_mean"] = statistics.mean([B[k]["lm_tokens"] for k in ks if k in B]) if any(k in B for k in ks) else None
        r["lm_tokens_C_mean"] = statistics.mean([C[k]["C"]["tokens"] for k in ks if k in C]) if any(k in C for k in ks) else None
        r["lm_tokens_Cd_mean"] = statistics.mean([C[k]["Cd"]["tokens"] for k in ks if k in C]) if any(k in C for k in ks) else None
        r["cpu_ms_C_mean"] = statistics.mean([C[k]["C"]["wall_ms"] for k in ks if k in C]) if any(k in C for k in ks) else None
        r["A_tokens_mean"] = statistics.mean([A[k]["completion_tokens"] or 0 for k in ks if k in A]) if any(k in A for k in ks) else None
        r["A_wall_s_mean"] = statistics.mean([A[k]["wall_s"] for k in ks if k in A]) if any(k in A for k in ks) else None
        return r

    held, near = arm("held"), arm("near")
    per_t = []
    for t in sel["templates"]:
        row = {"tid": t["tid"], "entry": t["tid"] in ents, "source_build_matches_golden": (ents[t["tid"]]["source_matches_golden"] if t["tid"] in ents else None)}
        for kind in ("held", "near"):
            ks = ["%s-%d-%02d" % (kind, x[1], x[2]) for x in t[kind]]
            row[kind] = {"n": len(ks), "A": sum(1 for k in ks if k in A and A[k]["correct"]), "hit": sum(1 for k in ks if hit(k)), "hit_right": sum(1 for k in ks if hit(k) and B[k]["correct"]),
                         "hit_wrong": sum(1 for k in ks if hit(k) and not B[k]["correct"])}
        per_t.append(row)
    card = {}
    cp = os.path.join(run, "card.json")
    if os.path.exists(cp):
        card = json.load(open(cp))
    st = {"n_held": n, "n_near": nn, "entries": len(ents), "entries_failed": nfail, "A_held_k": held["A_k"], "Bg_held_k": held["Bg_k"], "Crule_held_k": held["Crule_k"],
          "near_hit_wrong_k": near["hit_wrong_k"], "incomplete": incomplete, "build_complete": build_complete,
          "min_entries": (len(sel["templates"]) if sel.get("mock_subset") else MIN_ENTRIES)}
    v, holds, trap = decide(st)
    ent_gold = sum(1 for e in ents.values() if e["source_matches_golden"])
    res = {"verdict": v, "holds": holds, "trap": trap, "line1": line1(v, st), "counts": st, "threshold": {"tau": tau, "rule": "largest s keeping >= %d%% of held-out main as served hits" % int(HIT_FLOOR * 100),
           "floor_unreachable": unreachable, "served_held": len(served_s), "needed": need}, "held": held, "near": near, "per_template": per_t, "entries": {"validated": len(ents), "failed": nfail,
           "source_build_answer_equals_golden": ent_gold, "attempts_mean": (statistics.mean([len(e["attempts"]) for e in ents.values()]) if ents else None),
           "builder_tokens_mean": (statistics.mean([e["builder_tokens"] for e in ents.values()]) if ents else None)}, "card": card,
           "A_decode_tok_s_median": statistics.median([A[k]["decode_tok_s"] for k in A if A[k].get("decode_tok_s")]) if any(A[k].get("decode_tok_s") for k in A) else None}
    return res


def render_text(res):
    h, nr = res["held"], res["near"]

    def row(name, kk, n, ci):
        return "  %-34s %3d/%-3d  %5s%%  [%s-%s]" % (name, kk, n, pct(kk / max(1, n), 1), pct(ci[0], 1), pct(ci[1], 1))
    L = ["VERDICT %s | holds=%s trap=%s | entries %d/20 validated" % (res["verdict"], res["holds"], res["trap"], res["entries"]["validated"]), "LINE1 %s" % res["line1"], ""]
    L.append("threshold tau=%.4f (%s)%s; served held-out %d, needed %d" % (res["threshold"]["tau"], res["threshold"]["rule"], " FLOOR UNREACHABLE" if res["threshold"]["floor_unreachable"] else "", res["threshold"]["served_held"], res["threshold"]["needed"]))
    for nm, r in (("held-out main", h), ("near-miss p1", nr)):
        L += ["", nm + " (n=%d), accuracy with Wilson 95%%:" % r["n"], row("A source, from scratch", r["A_k"], r["n"], r["A_ci"]),
              row("B gated (cache, miss -> A)", r["Bg_k"], r["n"], r["Bg_ci"]), row("B cache-only (miss = wrong)", r["Bc_k"], r["n"], r["Bc_ci"]),
              row("C 1B alone, brief reasoning", r["C_k"], r["n"], wilson(r["C_k"], r["n"])), row("Cd 1B alone, number only", r["Cd_k"], r["n"], wilson(r["Cd_k"], r["n"])),
              row("cache hits", r["hit_k"], r["n"], r["hit_ci"]), row("cache hits that are WRONG", r["hit_wrong_k"], r["n"], r["hit_wrong_ci"]),
              "  McNemar exact A vs B gated: A-only-right %d, B-only-right %d, p=%.4f ; A vs B cache-only: %d / %d, p=%.4f" % (r["b_AonlyRight"], r["c_BonlyRight"], r["mcnemar_gated_p"], r["b_AonlyRight_cacheonly"], r["c_BonlyRight_cacheonly"], r["mcnemar_cacheonly_p"]),
              "  median s %.3f ; routed to own template's entry %s%% ; B cpu ms/query median %s mean %s ; 1B tokens/query B %s, C %s, Cd %s ; A tokens/query %s, wall %s s" % (
                  r["s_median"] or 0, pct(r["routing_to_own_entry"]), "%.0f" % r["cpu_ms_B_median"] if r["cpu_ms_B_median"] else "-", "%.0f" % r["cpu_ms_B_mean"] if r["cpu_ms_B_mean"] else "-",
                  "%.1f" % r["lm_tokens_B_mean"] if r["lm_tokens_B_mean"] is not None else "-", "%.0f" % r["lm_tokens_C_mean"] if r["lm_tokens_C_mean"] is not None else "-",
                  "%.1f" % r["lm_tokens_Cd_mean"] if r["lm_tokens_Cd_mean"] is not None else "-", "%.0f" % r["A_tokens_mean"] if r["A_tokens_mean"] else "-", "%.0f" % r["A_wall_s_mean"] if r["A_wall_s_mean"] else "-")]
    e = res["entries"]
    L += ["", "entries: %d validated, %d failed; the source's build-instance answer equalled the golden label on %d of %d (logged only, never used in the build loop); builder attempts mean %s, builder tokens mean %s" % (
        e["validated"], e["failed"], e["source_build_answer_equals_golden"], e["validated"], "%.2f" % e["attempts_mean"] if e["attempts_mean"] else "-", "%.0f" % e["builder_tokens_mean"] if e["builder_tokens_mean"] else "-")]
    if res["card"]:
        L.append("card: %s" % json.dumps(res["card"]))
    return "\n".join(L)


def main_run(run, selp, mock=False):
    sel = json.load(open(selp))
    res = compute(run, sel)
    os.makedirs(os.path.join(run, "out"), exist_ok=True)
    txt = render_text(res)
    open(os.path.join(run, "out", "summary.txt"), "w").write(txt + "\n")
    if res["verdict"] == "INCOMPLETE":
        json.dump(res, open(os.path.join(run, "summary.pre.json"), "w"), indent=1)
        print(txt)
        return 3
    if mock:
        res["MOCK"] = True
    json.dump(res, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print(txt)
    return 0


def selftest():
    base = {"n_held": 200, "n_near": 200, "entries": 20, "entries_failed": 0, "A_held_k": 170, "Bg_held_k": 168, "Crule_held_k": 60, "near_hit_wrong_k": 70, "incomplete": False}
    cases = [("HOLDS AND TRAPS", {}), ("HOLDS, NO TRAP", {"near_hit_wrong_k": 20}), ("DOES NOT HOLD, TRAPS", {"Bg_held_k": 150}), ("DOES NOT HOLD, NO TRAP", {"Bg_held_k": 150, "near_hit_wrong_k": 39}),
             ("DOES NOT HOLD, NO TRAP", {"Bg_held_k": 168, "Crule_held_k": 150, "near_hit_wrong_k": 0}), ("HOLDS AND TRAPS", {"Bg_held_k": 164, "near_hit_wrong_k": 40}),
             ("DOES NOT HOLD, TRAPS", {"Bg_held_k": 163, "near_hit_wrong_k": 40}), ("INCOMPLETE", {"entries": 9, "incomplete": True}), ("BROKEN", {"entries": 9, "build_complete": True, "incomplete": True}), ("BROKEN", {"A_held_k": 99}), ("INCOMPLETE", {"incomplete": True})]
    bad = 0
    for want, over in cases:
        st = dict(base); st.update(over)
        got = decide(st)[0]
        ok = got == want
        bad += (not ok)
        print("%-24s %-24s %s" % (want, got, "ok" if ok else "FAIL"), "|", line1(got, st)[:90])
    # boundary: exactly A-3 pts holds, exactly 20% trap fires
    s2 = dict(base); s2.update({"Bg_held_k": 164, "near_hit_wrong_k": 40}); assert decide(s2)[0] == "HOLDS AND TRAPS"
    print("selftest:", "FAIL %d" % bad if bad else "ok %d branches" % len(cases))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main_run(sys.argv[1], sys.argv[2], "--mock" in sys.argv))
