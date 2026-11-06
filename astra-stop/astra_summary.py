#!/usr/bin/env python3
"""astra_summary.py: applies the PRE-REGISTERED rule (PREREG.md) to results/<arm>/pelican-*.json (+ maths-*.json, reported only).
usage: astra_summary.py <run_dir> [--selftest]. Writes <run_dir>/summary.pre.json always; summary.json only when n=N_WANT for both arms.
Verdicts (overall): INVALID | BROKEN | INCOMPLETE | REPRODUCED | PARTIAL | NO DIFFERENCE | REVERSED."""
import glob, json, os, sys
import numpy as np
from scipy import stats

N_WANT = 32
ALPHA = 0.05
REPRO_MIN_MEDIAN_DROP = 0.25      # REPRODUCED needs the median thinking tokens down >= 25% AND p < ALPHA (the card claims -37% median / -40% mean)
BROKEN_DROP_PTS = 25.0            # edited SVG completeness more than 25 points below the unedited arm


def load(run, arm, kind):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "results", arm, f"{kind}-*.json"))):
        if f.endswith(".tok.json"):
            continue
        rows.append(json.load(open(f)))
    return rows


def boot_ratio(a, b, fn, n=10000, seed=7):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); b = np.asarray(b, float)
    r = [fn(rng.choice(b, len(b))) / fn(rng.choice(a, len(a))) - 1 for _ in range(n)]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def arm_stats(rows):
    t = np.array([r["think_tokens"] for r in rows], float)
    if len(t) == 0:
        return dict(n=0)
    comp = sum(1 for r in rows if r.get("complete"))
    return dict(n=len(t), mean=float(t.mean()), median=float(np.median(t)), q1=float(np.percentile(t, 25)), q3=float(np.percentile(t, 75)),
                min=float(t.min()), max=float(t.max()), under5k=int((t < 5000).sum()), capped=sum(1 for r in rows if r.get("finish_reason") == "length"),
                complete=comp, complete_pct=round(100 * comp / len(t), 1),
                wall_median_s=float(np.median([r["wall_s"] for r in rows])), tps_median=float(np.median([r["decode_tps_server"] or 0 for r in rows])))


def verdict(run):
    P = {a: load(run, a, "pelican") for a in ("base", "edited")}
    S = {a: arm_stats(P[a]) for a in P}
    fit = {}
    for a in ("base", "edited"):
        f = os.path.join(run, f"fit-{a}.json")
        fit[a] = json.load(open(f)) if os.path.exists(f) else None
    out = dict(arms=S, fit=fit, n_want=N_WANT, provisional=not all(S[a]["n"] >= N_WANT for a in S))
    why = []
    if fit["edited"] is not None and not fit["edited"].get("loaded"):
        out["verdict"] = "BROKEN"; why.append("edited arm did not load or failed the template smoke")
    elif fit["base"] is not None and not fit["base"].get("loaded"):
        out["verdict"] = "INVALID"; why.append("unedited arm did not load: no comparison")
    elif S["base"]["n"] == 0 or S["edited"]["n"] == 0:
        out["verdict"] = "INCOMPLETE"; why.append("an arm has no completed try")
    else:
        b = np.array([r["think_tokens"] for r in P["base"]], float); e = np.array([r["think_tokens"] for r in P["edited"]], float)
        mw = stats.mannwhitneyu(e, b, alternative="two-sided"); p = float(mw.pvalue)
        out["mannwhitney_p"] = p; out["rank_biserial_edited_vs_base"] = float(2 * mw.statistic / (len(e) * len(b)) - 1)
        out["median_change_pct"] = round(100 * (S["edited"]["median"] / S["base"]["median"] - 1), 1)
        out["mean_change_pct"] = round(100 * (S["edited"]["mean"] / S["base"]["mean"] - 1), 1)
        out["mean_change_ci95_pct"] = [round(100 * x, 1) for x in boot_ratio(b, e, np.mean)]
        out["median_change_ci95_pct"] = [round(100 * x, 1) for x in boot_ratio(b, e, np.median)]
        fe = stats.fisher_exact([[S["edited"]["complete"], S["edited"]["n"] - S["edited"]["complete"]], [S["base"]["complete"], S["base"]["n"] - S["base"]["complete"]]])
        out["completeness_fisher_p"] = float(fe[1])
        drop = S["base"]["complete_pct"] - S["edited"]["complete_pct"]
        out["completeness_drop_pts"] = round(drop, 1)
        if drop > BROKEN_DROP_PTS:
            out["verdict"] = "BROKEN"; why.append(f"edited SVG completeness {S['edited']['complete_pct']}% is {drop:.0f} points under the unedited {S['base']['complete_pct']}%")
        elif p < ALPHA and S["edited"]["median"] > S["base"]["median"]:
            out["verdict"] = "REVERSED"; why.append("edited arm thinks significantly LONGER")
        elif p < ALPHA and -out["median_change_pct"] >= 100 * REPRO_MIN_MEDIAN_DROP:
            out["verdict"] = "REPRODUCED"; why.append(f"median thinking tokens {out['median_change_pct']}%, p={p:.4f}")
        elif p < ALPHA and S["edited"]["median"] < S["base"]["median"]:
            out["verdict"] = "PARTIAL"; why.append(f"significant but median drop only {-out['median_change_pct']}% (< 25%)")
        else:
            out["verdict"] = "NO DIFFERENCE"; why.append(f"p={p:.3f}: no difference detected at n={S['base']['n']}/{S['edited']['n']} on the quantised pair (a Q4 non-survival is one reading, low power another)")
    if out["provisional"] and out["verdict"] not in ("BROKEN", "INVALID"):
        why.append(f"PROVISIONAL: n={S['base'].get('n',0)}/{S['edited'].get('n',0)} of {N_WANT}")
        out["verdict_if_final"] = out["verdict"]; out["verdict"] = "INCOMPLETE"
    out["why"] = why
    M = {a: load(run, a, "maths") for a in ("base", "edited")}
    out["maths_reported_only"] = {a: (dict(n=len(M[a]), correct=sum(1 for r in M[a] if r.get("correct")), think_mean=float(np.mean([r["think_tokens"] for r in M[a]])), think_median=float(np.median([r["think_tokens"] for r in M[a]])),
                                          capped=sum(1 for r in M[a] if r.get("finish_reason") == "length")) if M[a] else dict(n=0)) for a in M}
    return out


def line1(o):
    s = o["arms"]
    if o["verdict"] in ("REPRODUCED", "PARTIAL", "NO DIFFERENCE", "REVERSED"):
        return (f"on Q4_K_M the edited Qwen3.8-27B thought a median {s['edited']['median']:.0f} tokens on the pelican prompt vs {s['base']['median']:.0f} unedited "
                f"({o['median_change_pct']:+.0f}% median, {o['mean_change_pct']:+.0f}% mean, Mann-Whitney p={o['mannwhitney_p']:.3g}, n={s['edited']['n']}/{s['base']['n']}); "
                f"SVGs complete {s['edited']['complete_pct']:.0f}% vs {s['base']['complete_pct']:.0f}%.")
    return "no line 1: verdict " + o["verdict"]


def main():
    run = sys.argv[1]
    o = verdict(run)
    json.dump(o, open(os.path.join(run, "summary.pre.json"), "w"), indent=1)
    if o["verdict"] not in ("INCOMPLETE",):
        json.dump(o, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print("VERDICT", o["verdict"], "|", "; ".join(o["why"]))
    print("LINE1", line1(o))
    print(json.dumps({k: o[k] for k in o if k not in ("fit",)}, indent=1)[:2500])
    sys.exit(0 if o["verdict"] != "INCOMPLETE" else 3)


if __name__ == "__main__":
    main()
