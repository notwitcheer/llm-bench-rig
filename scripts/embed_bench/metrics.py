"""Pure ranking metrics with binary relevance. No numpy needed; deterministic."""
import math


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
