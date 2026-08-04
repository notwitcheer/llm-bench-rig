"""t090: materialise the task sets ONCE, to pinned JSONL with recorded hashes.

The quant tax is a within-model comparison across a quant ladder, so the single
thing that must never move is the input. Every quant sees byte-identical prompts
from these files; the sha256 in MANIFEST.json is what proves it afterwards.

Run (capsule): PYTHONPATH=. ~/sglang-env/bin/python scripts/quant_tax/build_datasets.py
"""

import argparse
import hashlib
import json
import os
import random

OUT = "dataset/quant_tax"

# Fixed MMLU subject set: four spread across the knowledge space, so a subject-level
# cliff is visible rather than averaged away.
MMLU_SUBJECTS = [
    "high_school_mathematics",
    "professional_medicine",
    "world_religions",
    "college_computer_science",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def build_gsm8k(n):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds.select(range(min(n, len(ds))))):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"id": f"gsm8k-{i}", "task": "gsm8k", "question": r["question"], "gold": gold})
    return write_jsonl("gsm8k", rows)


def build_mmlu(per_subject, seed):
    from datasets import load_dataset

    rows = []
    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test")
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        for i in idx[:per_subject]:
            r = ds[i]
            rows.append(
                {
                    "id": f"mmlu-{subj}-{i}",
                    "task": "mmlu",
                    "subject": subj,
                    "question": r["question"],
                    "choices": r["choices"],
                    "gold": "ABCD"[r["answer"]],
                }
            )
    return write_jsonl("mmlu", rows)


def build_humaneval():
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = [
        {
            "id": r["task_id"],
            "task": "humaneval",
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in ds
    ]
    return write_jsonl("humaneval", rows)


def build_wikitext(n_chars):
    """Raw text for llama-perplexity, which takes a file rather than a dataset."""
    from datasets import load_dataset

    # namespaced id: the bare "wikitext" alias no longer resolves on the hub
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(r["text"] for r in ds)[:n_chars]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/wikitext2_test.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k-n", type=int, default=250)
    ap.add_argument("--mmlu-per-subject", type=int, default=60)
    ap.add_argument("--wikitext-chars", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    paths = {
        "gsm8k": build_gsm8k(a.gsm8k_n),
        "mmlu": build_mmlu(a.mmlu_per_subject, a.seed),
        "humaneval": build_humaneval(),
        "wikitext_ppl": build_wikitext(a.wikitext_chars),
    }

    manifest = {
        "task": "t090",
        "seed": a.seed,
        "mmlu_subjects": MMLU_SUBJECTS,
        "sources": {
            "gsm8k": "openai/gsm8k:main:test",
            "mmlu": "cais/mmlu:<subject>:test",
            "humaneval": "openai/openai_humaneval:test",
            "wikitext_ppl": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        },
        "files": {
            k: {"path": p, "sha256": sha256(p), "bytes": os.path.getsize(p)}
            for k, p in paths.items()
        },
    }
    with open(f"{OUT}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for k, v in manifest["files"].items():
        print(f"{k:14s} {v['bytes']:>10,} B  {v['sha256'][:16]}  {v['path']}")
    print(f"\nmanifest -> {OUT}/MANIFEST.json")


if __name__ == "__main__":
    main()
