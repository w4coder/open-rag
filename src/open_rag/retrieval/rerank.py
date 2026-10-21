"""Reranking: re-score the fused candidates with a more precise (but slower) model.

WHY: bi-encoders embed query and doc separately (fast, approximate). A cross-encoder reads
them *together* (slow, accurate), so we apply it only to the ~20-40 fused candidates.
It is typically the best quality-per-effort upgrade after hybrid search.
"""
from __future__ import annotations

import logging
from typing import Protocol

from open_rag.config import Settings
from open_rag.generation.chains import build_llm_scorer
from open_rag.indexing.store import Hit

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class NoReranker:
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder  # optional extra: pip install .[rerank]

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)[:top_k]
        return [Hit(**{**h.__dict__, "score": float(s)}) for h, s in ranked]


class LLMReranker:
    def __init__(self, settings: Settings) -> None:
        self.scorer = build_llm_scorer(settings)
        self.min_relevance = settings.min_relevance

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits or self.scorer is None:
            return hits[:top_k]
        # Short numbers, not chunk UUIDs: the model copies "[7]" reliably, 36-char ids less so.
        passages = "\n".join(f"[{i}] {h.text[:600]}" for i, h in enumerate(hits, 1))
        try:
            out = self.scorer.invoke({"question": query, "passages": passages})
        except Exception:
            # Degrade, don't fail: reranking refines an already good hybrid ranking. Keep the
            # RRF order, with scores lifted to the relevance floor so the min_relevance filter
            # (calibrated on LLM scores) does not drop every passage.
            log.warning("LLM reranker failed; falling back to the hybrid ranking", exc_info=True)
            return [Hit(**{**h.__dict__, "score": max(h.score, self.min_relevance)})
                    for h in hits[:top_k]]
        by_pos = {s.id: s.score for s in out.scores}
        scored = [(h, by_pos.get(i, 0.0)) for i, h in enumerate(hits, 1)]
        ranked = sorted(scored, key=lambda p: p[1], reverse=True)[:top_k]  # stable: ties keep RRF order
        return [Hit(**{**h.__dict__, "score": s}) for h, s in ranked]


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker()
    if settings.reranker == "llm":
        return LLMReranker(settings)
    return NoReranker()
