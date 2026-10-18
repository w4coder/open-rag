"""Zero-downtime re-indexing (blue/green) with live writes.

Use when the embedding model, the sparse encoder or the collection schema changes: build a
*new* collection from the stored payloads, validate it, then swap the alias.

Correctness while ingestion keeps running:
1. **dual write** is switched on first: every live upsert/delete also lands in the new
   collection (VectorIndex._targets);
2. all payloads are copied (paginated over *every* tenant, `extra` fields such as the LLM
   context preserved, re-embedded with the current model);
3. **catch-up**: tenant versions are compared before/after each pass; a tenant written to
   during the copy may have received a stale copy, so it is re-copied until a pass sees no
   change (bounded);
4. **validation gate** (default: per-tenant point counts match; plug your eval suite here);
5. atomic alias swap (+ epoch bump => every cache entry is invalidated), dual write off.
Any failure aborts: dual write off, new collection dropped, live alias untouched.
Rollback after a swap = `rollback(service, old_collection)`.

A *chunking* change needs the source documents (kept on the upload volume / object
storage): re-run ingestion against the new collection instead.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from open_rag.indexing.store import VectorIndex
from open_rag.ingestion.chunking import Chunk

log = logging.getLogger(__name__)
_CORE = {"chunk_id", "text", "parent_text", "source", "tenant", "section", "doc_hash"}


def _chunk(p: dict) -> Chunk:
    return Chunk(id=p["chunk_id"], text=p["text"], parent_text=p.get("parent_text", p["text"]),
                 source=p["source"], tenant=p["tenant"], section=p.get("section", ""),
                 doc_hash=p.get("doc_hash", ""), extra={k: v for k, v in p.items() if k not in _CORE})


def _copy(index: VectorIndex, old: str, new: str, tenant: str | None = None,
          batch: int = 256) -> int:
    buf, total = [], 0
    for payload in index.scroll_payloads(old, tenant=tenant):
        buf.append(_chunk(payload))
        if len(buf) >= batch:
            total += index.upsert(buf, collection=new, bump=False, replace=False)
            buf = []
    return total + index.upsert(buf, collection=new, bump=False, replace=False)


def counts_match(index: VectorIndex, old: str, new: str, tenants: set[str]) -> bool:
    return all(index.count(t, collection=old) == index.count(t, collection=new) for t in tenants)


def reindex(service, validate: Callable[[str], bool] | None = None, max_passes: int = 5) -> str:
    index: VectorIndex = service.index
    old = index.collection
    new = index.new_version()
    index.set_dual_write(new)
    try:
        tenants = {p["tenant"] for p in index.scroll_payloads(old)}
        before = index.tenant_versions(tenants)
        total = _copy(index, old, new)
        for _ in range(max_passes):
            tenants |= {p["tenant"] for p in index.scroll_payloads(old)}
            after = index.tenant_versions(tenants)
            changed = {t for t in tenants if after[t] != before.get(t)}
            if not changed:
                break
            log.info("catch-up pass for tenants written during the copy: %s", sorted(changed))
            for t in changed:
                index.delete_tenant(t, new)
                _copy(index, old, new, tenant=t)
            before = after
        else:
            raise RuntimeError("index kept changing during re-index; retry when quieter")
        ok = validate(new) if validate else counts_match(index, old, new, tenants)
        if not ok:
            raise RuntimeError(f"validation failed for {new}: alias left on {old}")
        index.swap_alias(new)
    except BaseException:
        index.set_dual_write(None)
        index.drop_collection(new)
        raise
    index.set_dual_write(None)
    return f"re-indexed {total} chunks: {old} -> {new} (rollback: `cli rollback {old}`)"


def rollback(service, collection: str) -> str:
    index: VectorIndex = service.index
    if not index.client.collection_exists(collection):
        raise ValueError(f"unknown collection {collection}")
    current = index.collection
    index.swap_alias(collection)
    return f"alias {index.alias}: {current} -> {collection}"
