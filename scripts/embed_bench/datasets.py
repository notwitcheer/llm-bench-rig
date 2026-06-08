"""Load a BEIR dataset as (corpus, queries, qrels) via ir_datasets.

corpus:  {doc_id: text}
queries: {query_id: text}
qrels:   {query_id: set(relevant_doc_id)}   # binary relevance (relevance > 0)
"""
import ir_datasets

# BEIR subsets used by the bench (small, clean gold relevance)
BEIR = {
    "scifact": "beir/scifact/test",
    "nfcorpus": "beir/nfcorpus/test",
}


def load_beir(name):
    ds = ir_datasets.load(BEIR[name])
    corpus = {}
    for doc in ds.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        corpus[doc.doc_id] = (title + "\n" + text).strip()
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = {}
    for qr in ds.qrels_iter():
        if qr.relevance > 0:
            qrels.setdefault(qr.query_id, set()).add(qr.doc_id)
    return corpus, queries, qrels


if __name__ == "__main__":
    import sys
    c, q, r = load_beir(sys.argv[1] if len(sys.argv) > 1 else "scifact")
    print(f"corpus={len(c)} queries={len(q)} qrels={len(r)}")
