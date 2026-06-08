import numpy as np
import pytest
from scripts.embed_bench.retrieval import cosine_rank, truncate_dims


def test_cosine_rank_orders_by_similarity():
    query = np.array([1.0, 0.0])
    doc_vecs = np.array([
        [0.9, 0.1],   # dA, close to query
        [0.1, 0.9],   # dB, far
        [1.0, 0.0],   # dC, identical direction
    ])
    ids = ["dA", "dB", "dC"]
    assert cosine_rank(query, doc_vecs, ids) == ["dC", "dA", "dB"]


def test_cosine_rank_is_magnitude_invariant():
    # scaling a doc vector must not change its rank (cosine, not dot)
    query = np.array([1.0, 0.0])
    doc_vecs = np.array([[5.0, 0.0], [0.0, 1.0]])
    assert cosine_rank(query, doc_vecs, ["big", "orthogonal"])[0] == "big"


def test_truncate_dims_slices_and_renormalizes():
    vecs = np.array([[3.0, 4.0, 0.0, 0.0]])   # 4-dim, norm 5
    out = truncate_dims(vecs, 2)
    assert out.shape == (1, 2)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    # direction preserved: [3,4] normalized -> [0.6, 0.8]
    assert out[0] == pytest.approx([0.6, 0.8])


def test_truncate_dims_noop_when_dim_ge_native():
    vecs = np.array([[1.0, 0.0]])
    out = truncate_dims(vecs, 2)
    assert out.shape == (1, 2)
