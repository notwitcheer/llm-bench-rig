"""Aggregate t074 steering-arm eval results into a claimed-vs-measured table.
build_table is pure (tested); main() globs the graded results/em/*.json (grade_em.py
schema) plus results/steering/*train_log.json timing, prints the table, and dumps
results/steering/summary.json for chart_steering.py.

Run (Mac): venv/bin/python scripts/aggregate_steering.py
"""
import glob
import json
import os

from lib.steering import parse_timing, timing_summary


def build_table(rows: list[dict]) -> dict:
    """{benchmark: {label: {acc, delta_vs_base, rep_rate, mean_len}}} (deltas 1dp)."""
    out = {}
    for r in rows:
        out.setdefault(r["benchmark"], {})[r["label"]] = {
            "acc": r["acc"], "rep_rate": r["rep_rate"], "mean_len": r["mean_len"]}
    for bench, labels in out.items():
        base = labels.get("base", {}).get("acc")
        for label, rec in labels.items():
            rec["delta_vs_base"] = (round(rec["acc"] - base, 1)
                                    if base is not None else None)
    return out


def main():
    rows = []
    for p in sorted(glob.glob("results/em/*.json")):
        if p.endswith(".gens.json"):
            continue
        d = json.load(open(p))
        if d.get("variant") != "paper":
            continue
        if not (d["label"] in ("base", "base-toprefix")
                or d["label"].startswith(("steer", "lora"))):
            continue   # skip t073's em-* results sharing this directory
        rows.append({"label": d["label"], "benchmark": d["benchmark"], "acc": d["acc"],
                     "rep_rate": d["rep_rate"], "mean_len": d["mean_len"]})
    table = build_table(rows)

    timing = {}
    for p in sorted(glob.glob("results/steering/*train_log.json")):
        recs = json.load(open(p))
        name = os.path.basename(p).replace("_train_log.json", "").replace("train_log.json", "run")
        lines = [f"STEER-TIMING step={r['step']} rollout_s={r['rollout_s']:.1f} "
                 f"grade_s={r['grade_s']:.1f} update_s={r['update_s']:.1f}" for r in recs]
        timing[name] = timing_summary(parse_timing(lines))

    for bench, labels in table.items():
        print(f"\n== {bench} ==")
        for label in sorted(labels):
            r = labels[label]
            print(f"  {label:24s} acc={r['acc']:5.1f}  d={r['delta_vs_base']:+5.1f}  "
                  f"rep={r['rep_rate']:.3f}  len={r['mean_len']:.0f}")
    for name, t in timing.items():
        print(f"\n== timing: {name} ==  rollout {t['rollout_s']:.0f}s ({t['rollout_share']:.0%})"
              f"  grade {t['grade_s']:.0f}s ({t['grade_share']:.0%})"
              f"  update {t['update_s']:.0f}s ({t['update_share']:.0%})")

    os.makedirs("results/steering", exist_ok=True)
    json.dump({"table": table, "timing": timing},
              open("results/steering/summary.json", "w"), indent=2)
    print("\nwrote results/steering/summary.json")


if __name__ == "__main__":
    main()
