import math
import pytest
from scripts.embed_bench.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall_counts_relevant_in_topk():
    ranked = ["d1", "d2", "d3", "d4"]
    relevant = {"d1", "d3", "d5"}        # 3 relevant, 2 of them in top-3
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)


def test_recall_zero_when_no_relevant():
    assert recall_at_k(["d1", "d2"], set(), 2) == 0.0


def test_mrr_is_reciprocal_of_first_hit_rank():
    ranked = ["d2", "d1", "d3"]          # first relevant (d1) at rank 2
    assert mrr_at_k(ranked, {"d1"}, 10) == pytest.approx(0.5)


def test_mrr_zero_when_no_hit_in_topk():
    ranked = ["d2", "d3", "d4", "d1"]    # d1 at rank 4, k=3 -> miss
    assert mrr_at_k(ranked, {"d1"}, 3) == 0.0


def test_ndcg_binary_relevance():
    ranked = ["d1", "d2", "d3"]          # d1, d3 relevant
    relevant = {"d1", "d3"}
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(1.5 / 1.6309298, abs=1e-6)


def test_ndcg_zero_when_no_relevant():
    assert ndcg_at_k(["d1", "d2"], set(), 2) == 0.0
