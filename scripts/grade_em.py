"""Grade the gens-only JSON from eval_math.py with the vendored Qwen2.5-Math grader, and
compute collapse metrics. Runs under grader-env (sympy 1.12 + vendored latex2sympy2 1.9.0 +
antlr 4.11.1 + regex) — kept separate from the vLLM env on purpose.

Run (capsule): PYTHONPATH=. ~/grader-env/bin/python scripts/grade_em.py results/em/base__paper__math500.gens.json
"""
import argparse
import json

from lib.em import mean_len, repetition_rate
from lib.qwen_math_grader import extract_answer, grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gens", nargs="+", help="one or more *.gens.json files")
    a = ap.parse_args()
    for gpath in a.gens:
        d = json.load(open(gpath))
        recs, correct, toks = [], 0, []
        for g in d["gens"]:
            pred = extract_answer(g["text"])
            ok = grade(pred, g["gold"])
            correct += int(ok)
            toks.append(g["token_ids"])
            recs.append({"id": g["id"], "pred": pred, "gold": g["gold"], "ok": ok, "text": g["text"]})
        out = {"variant": d["variant"], "benchmark": d["benchmark"], "model": d["model"],
               "label": d["label"], "max_tokens": d.get("max_tokens"), "n": d["n"],
               "correct": correct, "acc": 100.0 * correct / d["n"],
               "mean_len": mean_len(toks),
               "rep_rate": sum(repetition_rate(t) for t in toks) / len(toks),
               "gens": recs}
        path = gpath.replace(".gens.json", ".json")
        json.dump(out, open(path, "w"))
        print(f"{out['label']}/{out['variant']}/{out['benchmark']}: acc={out['acc']:.1f}  "
              f"rep={out['rep_rate']:.3f}  len={out['mean_len']:.0f}  -> {path}")


if __name__ == "__main__":
    main()
