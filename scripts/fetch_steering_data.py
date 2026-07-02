"""Dump a fixed 512-problem training slice of DeepScaleR (their dataset) to JSONL.
Seeded shuffle -> SAME slice for every arm (t074). Their config pins
"agentica-org/deep_scale_r-preview-dataset" (canonical HF id below resolves the same).
Run once on capsule (network ON):
  ~/benchmark-rig/.venv/bin/python scripts/fetch_steering_data.py
"""
import argparse
import json
import os
import random

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agentica-org/DeepScaleR-Preview-Dataset")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="dataset/steering/train512.jsonl")
    a = ap.parse_args()
    ds = load_dataset(a.dataset, split="train")
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i in idx[: a.n]:
            r = ds[i]
            f.write(json.dumps({"id": i, "problem": r["problem"],
                                "answer": str(r["answer"])}) + "\n")
    print(f"wrote {a.n} rows -> {a.out}")


if __name__ == "__main__":
    main()
