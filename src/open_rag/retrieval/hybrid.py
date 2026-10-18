"""Hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion (RRF).

WHY RRF: dense cosine scores and BM25 scores live on incomparable scales. RRF fuses by
*rank* only (score = sum 1/(k + rank)), needs no tuning/normalisation and is the de-facto
default (Qdrant, Elastic, Weaviate, Azure AI Search all ship it).

The fusion runs inside Qdrant (`VectorIndex.hybrid_search`): one network round trip, no
client-side thread pool. `rrf_fuse` is kept as a reference implementation / for merging
results coming from *different* systems (e.g. docs + another search engine).
"""
from __future__ import annotations

from open_rag.config import Settings
from open_rag.indexing.store import Hit, VectorIndex


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, limit: int | None = None) -> list[Hit]:
    scores: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit.chunk_id, hit)
    fused = sorted(by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    out = [Hit(**{**h.__dict__, "score": scores[h.chunk_id]}) for h in fused]
    return out[:limit] if limit else out


class HybridRetriever:
    def __init__(self, index: VectorIndex, settings: Settings) -> None:
        self.index, self.settings = index, settings

    def retrieve(self, query: str, tenant: str, query_vector: list[float] | None = None
                 ) -> list[Hit]:
        return self.index.hybrid_search(query, tenant, query_vector=query_vector)

    def retrieve_dense_only(self, query: str, tenant: str) -> list[Hit]:
        """Baseline used by evals to quantify the gain of hybrid search."""
        return self.index.dense_search(query, tenant, self.settings.top_k_dense)

    def retrieve_sparse_only(self, query: str, tenant: str) -> list[Hit]:
        return self.index.sparse_search(query, tenant, self.settings.top_k_sparse)
