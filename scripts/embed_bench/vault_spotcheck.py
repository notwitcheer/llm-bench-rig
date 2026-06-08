"""Run the trio over Donald's real vault with hand-written paraphrased queries.
Qualitative: prints top-3 note titles per query per arm. Expected to NOT discriminate
much (the cp13 finding) — small corpus dominates. Vault: ~/.hermes/vault/**/*.md
"""
import glob
import os
import sys
from scripts.embed_bench import encode, retrieval

VAULT = os.path.expanduser("~/.hermes/vault")
# (query, the note slug we'd expect) — edit to match the live vault at run time
QUERIES = [
    "what is the current focus of the project",
    "what model does Donald run on",
    "how does the daily brief work",
    "what are the privacy rules for the vault",
    "who is the user",
    "what was decided about DeFi",
    "what is the local-AI pivot",
    "how is memory captured",
]


def load_vault():
    ids, texts = [], []
    for path in glob.glob(os.path.join(VAULT, "**", "*.md"), recursive=True):
        ids.append(os.path.relpath(path, VAULT))
        with open(path) as f:
            texts.append(f.read())
    return ids, texts


def main(arm):
    ids, texts = load_vault()
    model, cfg = encode.load(arm)
    dvec, _, _ = encode.encode_texts(model, texts, cfg["doc_prefix"], batch_size=cfg["batch_size"])
    qvec, _, _ = encode.encode_texts(model, QUERIES, cfg["query_prefix"], batch_size=cfg["batch_size"])
    print(f"=== {arm} over {len(ids)} vault notes ===")
    for i, q in enumerate(QUERIES):
        top3 = retrieval.cosine_rank(qvec[i], dvec, ids)[:3]
        print(f"  q: {q!r}\n     -> {top3}")


if __name__ == "__main__":
    main(sys.argv[1])
