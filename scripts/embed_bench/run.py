"""Encode a BEIR dataset with one arm, rank+score every query, write a results JSON."""
import json
import os
import sys
import numpy as np
from scripts.embed_bench import models, datasets, encode, retrieval, metrics

KS = [1, 5, 10]


def run(arm, dataset, out_dir):
    corpus, queries, qrels = datasets.load_beir(dataset)
    doc_ids = list(corpus)
    model, cfg = encode.load(arm)

    bs = cfg["batch_size"]
    doc_vecs, doc_dps, doc_peak = encode.encode_texts(
        model, [corpus[d] for d in doc_ids], cfg["doc_prefix"], batch_size=bs)
    q_ids = [q for q in queries if q in qrels]
    q_vecs, q_dps, _ = encode.encode_texts(
        model, [queries[q] for q in q_ids], cfg["query_prefix"], batch_size=bs)

    agg = {f"recall@{k}": [] for k in KS}
    agg["mrr@10"], agg["ndcg@10"] = [], []
    for i, qid in enumerate(q_ids):
        ranked = retrieval.cosine_rank(q_vecs[i], doc_vecs, doc_ids)
        rel = qrels[qid]
        for k in KS:
            agg[f"recall@{k}"].append(metrics.recall_at_k(ranked, rel, k))
        agg["mrr@10"].append(metrics.mrr_at_k(ranked, rel, 10))
        agg["ndcg@10"].append(metrics.ndcg_at_k(ranked, rel, 10))

    result = {
        "arm": arm, "hf_id": cfg["hf_id"], "dataset": dataset,
        "n_docs": len(doc_ids), "n_queries": len(q_ids), "dim": int(doc_vecs.shape[1]),
        "doc_encode_dps": round(doc_dps, 1), "query_encode_dps": round(q_dps, 1),
        "peak_vram_gb": round(doc_peak, 2),
        "scores": {m: round(float(np.mean(v)), 4) for m, v in agg.items()},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{arm}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["scores"]), "->", path)
    return result


if __name__ == "__main__":
    arm, dataset = sys.argv[1], sys.argv[2]
    run(arm, dataset, f"results/embed-{dataset}")
