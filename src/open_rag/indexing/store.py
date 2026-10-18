"""Vector index (Qdrant): dense + BM25 sparse vectors, tenant filtering, versioning, aliases.

Key production ideas shown here:
* **Alias indirection** - every read/write targets alias `docs`; Qdrant resolves it to the
  physical collection `docs_vN`. A re-embedding builds `docs_v{N+1}` then atomically swaps the
  alias (see reindex.py). Rollback = swap back.
* **Hybrid search server-side** - one `query_points` call: dense prefetch + sparse (BM25)
  prefetch fused with RRF inside Qdrant. No in-process index, nothing to rebuild or go stale.
* **Index versioning** - a tiny `docs_meta` collection holds a random version token per
  tenant, rotated on every write, plus a global epoch rotated on every alias swap. Caches key
  on it, so *any* content change (even same chunk count) invalidates them, on every replica.
* **Crash-safe upserts** - embed first (failure => index untouched), upsert the new points
  (deterministic ids), *then* delete the document's stale points. Readers never see a
  document disappear.
* **Multi-tenancy by payload filter** - one collection, indexed `tenant` field. Fine up to
  many thousands of tenants; use Qdrant custom sharding by tenant beyond that.
"""
from __future__ import annotations

import logging
import re
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from open_rag.config import Settings
from open_rag.ingestion.chunking import Chunk
from open_rag.retrieval.sparse import sparse_doc_vector, sparse_query_vector

log = logging.getLogger(__name__)
DENSE, SPARSE = "dense", "bm25"


@dataclass
class Hit:
    chunk_id: str
    text: str
    parent_text: str
    source: str
    section: str
    score: float
    tenant: str = ""


def _hit_from_payload(payload: dict, score: float) -> Hit:
    return Hit(
        chunk_id=payload["chunk_id"],
        text=payload["text"],
        parent_text=payload.get("parent_text", payload["text"]),
        source=payload["source"],
        section=payload.get("section", ""),
        score=score,
        tenant=payload.get("tenant", ""),
    )


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _tenant_filter(tenant: str) -> models.Filter:
    return models.Filter(must=[_match("tenant", tenant)])


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
                            timeout=10)
    return QdrantClient(path=settings.qdrant_path)


class LegacySchemaError(RuntimeError):
    pass


class VectorIndex:
    def __init__(self, settings: Settings, embeddings: Embeddings, client: QdrantClient | None = None):
        self.settings = settings
        self.embeddings = embeddings
        self.client = client if client is not None else make_client(settings)
        self.alias = settings.collection_alias
        self.meta = f"{self.alias}_meta"
        if not self.client.collection_exists(self.meta):
            self.client.create_collection(self.meta, vectors_config={})
        self._ensure_initial_collection()

    # ---- metadata: versions + reindex state ---------------------------------------------
    def _meta_id(self, key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.alias}:{key}"))

    def _meta_get(self, *keys: str) -> dict[str, dict]:
        ids = {self._meta_id(k): k for k in keys}
        points = self.client.retrieve(self.meta, ids=list(ids), with_payload=True)
        return {ids[str(p.id)]: p.payload or {} for p in points}

    def _meta_put(self, key: str, payload: dict) -> None:
        self.client.upsert(self.meta, points=[
            models.PointStruct(id=self._meta_id(key), vector={}, payload=payload)])

    def bump_version(self, tenants) -> None:
        for t in set(tenants):
            self._meta_put(f"tenant:{t}", {"v": uuid.uuid4().hex})

    def tenant_versions(self, tenants) -> dict[str, str]:
        got = self._meta_get(*(f"tenant:{t}" for t in tenants))
        return {t: got.get(f"tenant:{t}", {}).get("v", "0") for t in tenants}

    def index_version(self, tenant: str) -> str:
        """Opaque token that changes whenever the tenant's searchable content can change."""
        got = self._meta_get("epoch", f"tenant:{tenant}")
        return f"{got.get('epoch', {}).get('v', '0')}:{got.get(f'tenant:{tenant}', {}).get('v', '0')}"

    def _dual_write_targets(self) -> list[str]:
        target = self._meta_get("reindex").get("reindex", {}).get("target")
        return [target] if target else []

    def set_dual_write(self, target: str | None) -> None:
        self._meta_put("reindex", {"target": target} if target else {})

    # ---- collection / alias management -------------------------------------------------
    def _alias_target(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def _create_collection(self, name: str) -> None:
        dim = len(self.embeddings.embed_query("dimension probe"))
        self.client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # IDF computed by Qdrant from live statistics -> BM25 without a side index.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Index the filter fields: mandatory for fast filtered search at scale.
        for field in ("tenant", "source"):
            with warnings.catch_warnings():  # local mode ignores payload indexes (expected)
                warnings.simplefilter("ignore")
                self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)

    def _ensure_initial_collection(self) -> None:
        if self._alias_target() is None:
            name = f"{self.alias}_v1"
            if not self.client.collection_exists(name):
                self._create_collection(name)
            self._point_alias(name)
        vectors = self.client.get_collection(self.collection).config.params.vectors
        self.legacy = not (isinstance(vectors, dict) and DENSE in vectors)
        if self.legacy:
            log.error("collection %s uses the legacy single-vector schema: run "
                      "`python -m open_rag.cli reindex` to migrate", self.collection)

    def _point_alias(self, collection: str) -> None:
        ops = []
        if self._alias_target() is not None:
            ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self.alias))
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=self.alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)  # atomic on server

    @property
    def collection(self) -> str:
        """Physical collection behind the alias (control-plane only: never on the hot path)."""
        return self._alias_target() or f"{self.alias}_v1"

    def new_version(self) -> str:
        """Create the next physical collection (e.g. docs_v2) for a blue/green re-index."""
        m = re.search(r"_v(\d+)$", self.collection)
        n = int(m.group(1)) + 1 if m else 2
        while self.client.collection_exists(f"{self.alias}_v{n}"):
            n += 1
        name = f"{self.alias}_v{n}"
        self._create_collection(name)
        return name

    def swap_alias(self, collection: str) -> None:
        self._point_alias(collection)
        self._meta_put("epoch", {"v": uuid.uuid4().hex})  # invalidates every cache entry
        self.legacy = False

    def drop_collection(self, name: str) -> None:
        if name == self.collection:
            raise ValueError("refusing to drop the live collection")
        self.client.delete_collection(name)

    # ---- write path --------------------------------------------------------------------
    def _points(self, chunks: list[Chunk], batch: int) -> list[models.PointStruct]:
        points = []
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            dense = self.embeddings.embed_documents([c.embed_text for c in part])
            points.extend(
                models.PointStruct(id=c.id, payload=c.payload(),
                                   vector={DENSE: v, SPARSE: sparse_doc_vector(c.embed_text)})
                for c, v in zip(part, dense)
            )
        return points

    def _targets(self, collection: str | None) -> list[str]:
        return [collection] if collection else [self.alias, *self._dual_write_targets()]

    def upsert(self, chunks: list[Chunk], collection: str | None = None, batch: int = 64,
               bump: bool = True, replace: bool = True) -> int:
        """Replace each (tenant, source) document by `chunks` without a visibility gap.

        `replace=False` only adds/overwrites points (bulk copy where one document's chunks
        may span several calls)."""
        if not chunks:
            return 0
        targets = self._targets(collection)
        groups: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for c in chunks:
            groups[(c.tenant, c.source)].append(c)
        for (tenant, source), group in groups.items():
            points = self._points(group, batch)  # may fail: nothing written yet
            keep = [p.id for p in points]
            for target in targets:
                for i in range(0, len(points), batch):
                    self.client.upsert(target, points=points[i : i + batch], wait=True)
                if replace:
                    self._delete(target, tenant, source, keep_ids=keep)
        if bump:
            self.bump_version(t for t, _ in groups)
        return len(chunks)

    def _delete(self, target: str, tenant: str, source: str | None,
                keep_ids: list[str] | None = None) -> None:
        must = [_match("tenant", tenant)] + ([_match("source", source)] if source else [])
        must_not = [models.HasIdCondition(has_id=keep_ids)] if keep_ids else []
        self.client.delete(target, wait=True, points_selector=models.FilterSelector(
            filter=models.Filter(must=must, must_not=must_not)))

    def delete_source(self, tenant: str, source: str, collection: str | None = None) -> None:
        for target in self._targets(collection):
            self._delete(target, tenant, source)
        self.bump_version([tenant])

    def delete_tenant(self, tenant: str, collection: str) -> None:
        self._delete(collection, tenant, None)

    # ---- read path ---------------------------------------------------------------------
    def _check(self) -> None:
        if self.legacy:
            raise LegacySchemaError("legacy collection schema: run `open_rag.cli reindex`")

    def hybrid_search(self, query: str, tenant: str, query_vector: list[float] | None = None
                      ) -> list[Hit]:
        """Dense + BM25 candidates fused with RRF, in a single Qdrant round trip."""
        self._check()
        s, flt = self.settings, _tenant_filter(tenant)
        vec = query_vector if query_vector is not None else self.embeddings.embed_query(query)
        prefetch = [models.Prefetch(query=vec, using=DENSE, limit=s.top_k_dense, filter=flt)]
        sparse = sparse_query_vector(query)
        if sparse.indices:
            prefetch.append(models.Prefetch(query=sparse, using=SPARSE, limit=s.top_k_sparse,
                                            filter=flt))
        res = self.client.query_points(
            self.alias, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=s.rrf_k)),
            query_filter=flt, limit=s.top_k_dense + s.top_k_sparse, with_payload=True,
        )
        return [_hit_from_payload(p.payload, p.score) for p in res.points]

    def dense_search(self, query: str, tenant: str, k: int,
                     query_vector: list[float] | None = None) -> list[Hit]:
        self._check()
        vec = query_vector if query_vector is not None else self.embeddings.embed_query(query)
        res = self.client.query_points(self.alias, query=vec, using=DENSE, limit=k,
                                       query_filter=_tenant_filter(tenant), with_payload=True)
        return [_hit_from_payload(p.payload, p.score) for p in res.points]

    def sparse_search(self, query: str, tenant: str, k: int) -> list[Hit]:
        self._check()
        sparse = sparse_query_vector(query)
        if not sparse.indices:
            return []
        res = self.client.query_points(self.alias, query=sparse, using=SPARSE, limit=k,
                                       query_filter=_tenant_filter(tenant), with_payload=True)
        return [_hit_from_payload(p.payload, p.score) for p in res.points]

    def count(self, tenant: str | None = None, collection: str | None = None) -> int:
        flt = _tenant_filter(tenant) if tenant else None
        return self.client.count(collection or self.alias, count_filter=flt, exact=True).count

    def scroll_payloads(self, collection: str | None = None, tenant: str | None = None
                        ) -> Iterator[dict]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection or self.alias, limit=256, offset=offset, with_payload=True,
                with_vectors=False, scroll_filter=_tenant_filter(tenant) if tenant else None,
            )
            for p in points:
                yield p.payload
            if offset is None:
                return

    def scan(self, tenant: str) -> list[Hit]:
        return [_hit_from_payload(p, 0.0) for p in self.scroll_payloads(tenant=tenant)]

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
