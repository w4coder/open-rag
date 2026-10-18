"""Sparse (lexical / BM25) encoding for Qdrant sparse vectors.

WHY hybrid: dense embeddings capture meaning but miss exact tokens (error codes, SKUs,
function names, acronyms). BM25 nails those. Combining both is the single most robust
retrieval improvement over dense-only.

HOW: BM25 = IDF(term) x saturated term frequency. We compute the TF part client-side and store
it as a sparse vector; Qdrant applies the IDF part server-side (`Modifier.IDF`) using live
collection statistics. Consequences:
* the lexical index lives *in* the vector store: shared by all replicas, updated atomically
  with the dense vector, never stale, no rebuild, no RAM per process;
* hybrid fusion (RRF) runs server-side in one round trip (see VectorIndex.hybrid_search).
"""
from __future__ import annotations

from collections import Counter

from qdrant_client import models

from open_rag.providers import token_hash, tokenize

K1, B, AVG_LEN = 1.2, 0.75, 256  # standard BM25 constants; AVG_LEN as in fastembed's BM25
_MOD = 2**31 - 1  # Qdrant sparse indices are uint32


def _index(tok: str) -> int:
    return token_hash(tok) % _MOD


def sparse_doc_vector(text: str) -> models.SparseVector:
    toks = tokenize(text)
    tf = Counter(_index(t) for t in toks)
    norm = K1 * (1 - B + B * len(toks) / AVG_LEN)
    items = sorted((i, f * (K1 + 1) / (f + norm)) for i, f in tf.items())
    return models.SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])


def sparse_query_vector(text: str) -> models.SparseVector:
    idx = sorted({_index(t) for t in tokenize(text)})
    return models.SparseVector(indices=idx, values=[1.0] * len(idx))
