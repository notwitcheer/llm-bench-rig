"""t090: grade the gens JSON. No GPU, no server.

Writes one score row per (quant, task), plus MMLU per-subject accuracy, which is
where a knowledge-vs-reasoning split would show up.

Run: PYTHONPATH=. ~/sglang-env/bin/python scripts/quant_tax/grade.py results/quant_tax/*.gens.json
"""

import argparse
import collections
import json
import os

from lib.quanttax import grade

DATA = "dataset/quant_tax"
OUT = "results/quant_tax"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gens", nargs="+")
    a = ap.parse_args()

    rows_by_task = {}
    scored = []
    for gpath in a.gens:
        d = json.load(open(gpath))
        task = d["task"]
        if task not in rows_by_task:
            rows_by_task[task] = {
                json.loads(l)["id"]: json.loads(l)
                for l in open(f"{DATA}/{task}.jsonl")
                if l.strip()
            }
        rows = rows_by_task[task]

        correct = 0
        by_subject = collections.defaultdict(lambda: [0, 0])
        empty = 0
        how_counts = collections.Counter()
        truncated = 0
        for g in d["gens"]:
            row = rows[g["id"]]
            ok, how = grade(row, g["text"])
            correct += int(ok)
            how_counts[how] += 1
            if (g.get("meta") or {}).get("finish_reason") == "length":
                truncated += 1
            if not g["text"].strip():
                empty += 1
            if task == "mmlu":
                s = by_subject[row["subject"]]
                s[0] += int(ok)
                s[1] += 1

        rec = {
            "quant": d["quant"],
            "task": task,
            "n": d["n"],
            "correct": correct,
            "accuracy": round(correct / d["n"], 4),
            "empty_outputs": empty,
            "truncated": truncated,
            "graded_by": dict(how_counts),
            "wall_s": d.get("wall_s"),
        }
        if by_subject:
            rec["by_subject"] = {
                k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
                for k, v in sorted(by_subject.items())
            }
        scored.append(rec)
        print(f"{d['quant']:8s} {task:10s} {correct:4d}/{d['n']:<4d} = {rec['accuracy']:.4f}"
              + (f"  (empty: {empty})" if empty else ""))

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/scores.json"
    existing = json.load(open(path)) if os.path.exists(path) else []
    keyed = {(r["quant"], r["task"]): r for r in existing}
    keyed.update({(r["quant"], r["task"]): r for r in scored})
    with open(path, "w") as f:
        json.dump(sorted(keyed.values(), key=lambda r: (r["task"], r["quant"])), f, indent=2)
    print(f"\nscores -> {path}")


if __name__ == "__main__":
    main()
