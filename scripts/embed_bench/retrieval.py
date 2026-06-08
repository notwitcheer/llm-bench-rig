"""Pure cosine retrieval + Matryoshka dimension truncation. numpy only."""
import numpy as np


def _l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def cosine_rank(query_vec, doc_vecs, doc_ids):
    """Return doc_ids sorted by descending cosine similarity to query_vec."""
    q = _l2_normalize(np.asarray(query_vec, dtype=np.float64).reshape(1, -1))[0]
    docs = _l2_normalize(np.asarray(doc_vecs, dtype=np.float64))
    sims = docs @ q
    order = np.argsort(-sims, kind="stable")
    return [doc_ids[i] for i in order]


def truncate_dims(vecs, dim):
    """Matryoshka: keep the first `dim` dims and L2-renormalize each row."""
    sliced = np.asarray(vecs, dtype=np.float64)[:, :dim]
    return _l2_normalize(sliced)
