"""Composition root: wires settings -> index -> retriever -> graph -> cache -> limits.

One `RagService` per process; it is stateless w.r.t. requests so N replicas can run behind
a load balancer (state lives in Qdrant / Redis).

`query()` and `stream()` share one execution path (`_execute`), so caching, budgets,
metrics and tracing behave identically whether the client streams or not.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from langchain_core.callbacks import UsageMetadataCallbackHandler

from open_rag.cache import SemanticCache
from open_rag.config import Settings, get_settings
from open_rag.generation.guardrails import redact_pii
from open_rag.graph import build_graph
from open_rag.indexing.store import VectorIndex
from open_rag.limits import Limits
from open_rag.models import QueryRequest, QueryResponse
from open_rag.observability import CACHE, LATENCY, REQUESTS, record_usage, request_id
from open_rag.providers import get_embeddings
from open_rag.retrieval.hybrid import HybridRetriever
from open_rag.retrieval.rerank import get_reranker

log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """A dependency (LLM provider, vector store) failed after retries and fallbacks."""


class BudgetExceeded(RuntimeError):
    """The tenant spent its daily LLM token budget."""


class RagService:
    def __init__(self, settings: Settings | None = None, index: VectorIndex | None = None,
                 limits: Limits | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings = get_embeddings(self.settings)
        self.index = index or VectorIndex(self.settings, self.embeddings)
        self.retriever = HybridRetriever(self.index, self.settings)
        self.reranker = get_reranker(self.settings)  # exposed for evals (retrieval ablation)
        self.graph = build_graph(self.settings, self.retriever, self.reranker)
        s = self.settings
        self.cache = SemanticCache(
            self.index.client, f"{s.collection_alias}_semcache",
            dim=len(self.embeddings.embed_query("dimension probe")),
            threshold=s.cache_threshold, ttl_s=s.cache_ttl_s,
        ) if s.cache_enabled else None
        self.limits = limits or Limits.from_settings(s)

    def index_version(self, tenant: str) -> str:
        return self.index.index_version(tenant)

    @staticmethod
    def _response(state: dict, started: float) -> QueryResponse:
        return QueryResponse(
            answer=state["answer"],
            citations=state.get("citations", []),
            grounded=state.get("grounded", False),
            trace={
                "status": state.get("status"),
                "reason": state.get("reason"),
                "search_query": state.get("search_query"),
                "n_candidates": len(state.get("candidates", [])),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def _execute(self, req: QueryRequest, tenant: str) -> Iterator[tuple[str, object]]:
        """Yields ("step", node_name)* then ("result", QueryResponse)."""
        if not self.limits.has_token_budget(tenant):
            raise BudgetExceeded(f"daily token budget exhausted for tenant {tenant}")
        started = time.perf_counter()
        history = [m.model_dump() for m in req.history]
        version = self.index_version(tenant)
        vector = None
        if not history:  # follow-ups depend on history: never serve them from cache
            vector = self.embeddings.embed_query(req.question)  # reused by retrieval
            if self.cache is not None:
                hit = self.cache.get(vector, tenant, version)
                CACHE.labels(result="hit" if hit else "miss").inc()
                if hit is not None:
                    REQUESTS.labels(status="cached").inc()
                    yield "step", "cache"
                    yield "result", hit
                    return
        usage = UsageMetadataCallbackHandler()
        # LangSmith: tags/metadata make traces filterable (by tenant, index version, request).
        # The question is PII-redacted before it goes into metadata.
        config = {
            "run_name": "rag_query",
            "tags": ["open-rag", f"tenant:{tenant}"],
            "metadata": {"index_version": version, "request_id": request_id.get(),
                         "question_redacted": redact_pii(req.question)},
            "callbacks": [usage],
        }
        state: dict = {"question": req.question, "history": history, "tenant": tenant,
                       "query_vector": vector}
        try:
            for update in self.graph.stream(state, config=config, stream_mode="updates"):
                for node, out in update.items():
                    state.update(out or {})
                    yield "step", node
        except Exception as e:
            REQUESTS.labels(status="error").inc()
            log.exception("pipeline failed tenant=%s", tenant)
            raise UpstreamError(str(e)) from e
        finally:
            self.limits.spend_tokens(tenant, record_usage(tenant, usage.usage_metadata))
        resp = self._response(state, started)
        status = state.get("status") or "unknown"
        REQUESTS.labels(status=status).inc()
        LATENCY.observe(time.perf_counter() - started)
        if status == "ok" and vector is not None and self.cache is not None:
            self.cache.put(vector, tenant, version, resp)
        log.info("query tenant=%s status=%s latency_ms=%s", tenant, status,
                 resp.trace["latency_ms"])
        yield "result", resp

    def query(self, req: QueryRequest, tenant: str) -> QueryResponse:
        for kind, value in self._execute(req, tenant):
            if kind == "result":
                return value  # type: ignore[return-value]
        raise RuntimeError("pipeline produced no result")

    def stream(self, req: QueryRequest, tenant: str) -> Iterator[tuple[str, object]]:
        return self._execute(req, tenant)
