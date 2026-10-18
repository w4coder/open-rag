"""Semantic cache: skip the whole pipeline for repeated / near-identical questions.

Key design points (the classic pitfalls):
* stored in Qdrant (`docs_semcache`): shared by every replica, ANN lookup instead of a linear
  scan, survives restarts;
* scoped by tenant (never leak answers across tenants);
* scoped by *index version* (`VectorIndex.index_version`: rotated on every write and alias
  swap) so any content change invalidates entries - even one that keeps the chunk count;
* similarity threshold high (0.97) - a wrong cache hit is worse than a miss;
* TTL on read + periodic purge; only grounded answers are cached;
* the caller passes the query vector it already computed: no extra embedding call;
* failures are logged and treated as a miss: the cache can never break a request.
"""
from __future__ import annotations

import logging
import time
import uuid
import warnings

from qdrant_client import QdrantClient, models

from open_rag.models import QueryResponse

log = logging.getLogger(__name__)


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class SemanticCache:
    def __init__(self, client: QdrantClient, name: str, dim: int, threshold: float = 0.97,
                 ttl_s: float = 3600, purge_every: int = 200) -> None:
        self.client, self.name = client, name
        self.threshold, self.ttl_s, self.purge_every = threshold, ttl_s, purge_every
        self._puts = 0
        self._ensure(dim)

    def _ensure(self, dim: int) -> None:
        if self.client.collection_exists(self.name):
            size = self.client.get_collection(self.name).config.params.vectors.size
            if size == dim:
                return
            self.client.delete_collection(self.name)  # embedding model changed: start fresh
        self.client.create_collection(
            self.name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for field in ("tenant", "version"):
                self.client.create_payload_index(self.name, field, models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.name, "ts", models.PayloadSchemaType.FLOAT)

    def get(self, vector: list[float], tenant: str, version: str) -> QueryResponse | None:
        try:
            res = self.client.query_points(
                self.name, query=vector, limit=1, score_threshold=self.threshold,
                with_payload=True, query_filter=models.Filter(must=[
                    _match("tenant", tenant), _match("version", version),
                    models.FieldCondition(key="ts", range=models.Range(gte=time.time() - self.ttl_s)),
                ]),
            )
        except Exception:
            log.warning("semantic cache lookup failed (treated as miss)", exc_info=True)
            return None
        if not res.points:
            return None
        resp = QueryResponse.model_validate_json(res.points[0].payload["response"])
        return resp.model_copy(update={"cached": True})

    def put(self, vector: list[float], tenant: str, version: str, resp: QueryResponse) -> None:
        if not resp.grounded:  # never cache guardrail failures
            return
        try:
            self.client.upsert(self.name, points=[models.PointStruct(
                id=uuid.uuid4().hex, vector=vector,
                payload={"tenant": tenant, "version": version, "ts": time.time(),
                         "response": resp.model_dump_json()})])
            self._puts += 1
            if self._puts % self.purge_every == 0:
                self.purge_expired()
        except Exception:
            log.warning("semantic cache write failed", exc_info=True)

    def purge_expired(self) -> None:
        self.client.delete(self.name, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="ts", range=models.Range(lt=time.time() - self.ttl_s))])))
