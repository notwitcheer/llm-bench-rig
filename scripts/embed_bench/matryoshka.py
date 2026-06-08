"""Re-score the two Qwen arms at truncated dims to plot quality-vs-vector-size."""
import json
import os
import sys
import numpy as np
from scripts.embed_bench import datasets, encode, retrieval, metrics

DIMS = [2048, 1024, 512, 256, 128]


def sweep(arm, dataset, out_dir):
    corpus, queries, qrels = datasets.load_beir(dataset)
    doc_ids = list(corpus)
    model, cfg = encode.load(arm)
    bs = cfg["batch_size"]
    doc_full, _, _ = encode.encode_texts(model, [corpus[d] for d in doc_ids], cfg["doc_prefix"], batch_size=bs)
    q_ids = [q for q in queries if q in qrels]
    q_full, _, _ = encode.encode_texts(model, [queries[q] for q in q_ids], cfg["query_prefix"], batch_size=bs)

    out = {}
    for dim in [d for d in DIMS if d <= cfg["native_dim"]]:
        dvec = retrieval.truncate_dims(doc_full, dim)
        qvec = retrieval.truncate_dims(q_full, dim)
        ndcgs = []
        for i, qid in enumerate(q_ids):
            ranked = retrieval.cosine_rank(qvec[i], dvec, doc_ids)
            ndcgs.append(metrics.ndcg_at_k(ranked, qrels[qid], 10))
        out[str(dim)] = round(float(np.mean(ndcgs)), 4)
        print(f"{arm} dim={dim} ndcg@10={out[str(dim)]}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{arm}-matryoshka.json"), "w") as f:
        json.dump({"arm": arm, "dataset": dataset, "ndcg_by_dim": out}, f, indent=2)


if __name__ == "__main__":
    sweep(sys.argv[1], sys.argv[2], f"results/embed-{sys.argv[2]}")
