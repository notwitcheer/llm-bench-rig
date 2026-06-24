"""Materialize the t073 eval sets + the single EM training example from the one-shot-em repo
(the exact data the paper evaluated on — faithful, and avoids ambiguous HF dataset ids).

  math500 -> Qwen2.5-Eval/evaluation/data/math500/test.jsonl   (problem, answer)  [500]
  amc23   -> Qwen2.5-Eval/evaluation/data/amc23/test.jsonl      (problem, answer)  [40]
  pi1     -> dataset/1shot_rlvr/pi1_r1280.parquet               (the single pi1 problem, repeated)

Usage: python scripts/fetch_em_data.py --repo third_party/one-shot-em
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a one-shot-em checkout")
    a = ap.parse_args()
    ev = os.path.join(a.repo, "Qwen2.5-Eval/evaluation/data")
    os.makedirs("dataset/em", exist_ok=True)

    def read_jsonl(p):
        return [json.loads(l) for l in open(p) if l.strip()]

    def dump(name, rows):
        with open(f"dataset/em/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return len(rows)

    m = read_jsonl(os.path.join(ev, "math500/test.jsonl"))
    n_m = dump("math500", [{"id": f"math500-{i}", "problem": r["problem"], "answer": str(r["answer"])}
                           for i, r in enumerate(m)])

    amc = read_jsonl(os.path.join(ev, "amc23/test.jsonl"))
    n_a = dump("amc23", [{"id": f"amc23-{i}", "problem": r["problem"], "answer": str(r["answer"])}
                         for i, r in enumerate(amc)])

    import pandas as pd
    pi = pd.read_parquet(os.path.join(a.repo, "dataset/1shot_rlvr/pi1_r1280.parquet"))
    n_p = dump("pi1", [{"problem": str(pi.iloc[0]["problem"])}])

    print(f"wrote dataset/em/: math500={n_m} amc23={n_a} pi1={n_p}")


if __name__ == "__main__":
    main()
