"""Ingestion pipeline: RawDocument -> chunks -> (optional LLM context) -> vector index.

Run synchronously (CLI/tests) or from a queue worker (see open_rag.queue). The function
is idempotent: the same document ingested twice produces the same state.
"""
from __future__ import annotations

import logging
from pathlib import Path

from open_rag.config import Settings
from open_rag.generation.chains import build_contextualizer
from open_rag.indexing.store import VectorIndex
from open_rag.ingestion.chunking import Chunk, chunk_document
from open_rag.ingestion.loaders import RawDocument, iter_corpus, load_file

log = logging.getLogger(__name__)


def enrich_chunks(doc: RawDocument, chunks: list[Chunk], settings: Settings) -> list[Chunk]:
    """Contextual Retrieval (Anthropic, 2024): prepend a short LLM-written context to each
    chunk *before embedding* so isolated chunks ('revenue grew 3%') keep their referent.
    Costly (1 LLM call/chunk): use prompt caching on the document + batch/async.
    """
    ctx = build_contextualizer(settings)
    if ctx is None:
        return chunks
    contexts = ctx.batch(
        [{"document": doc.text[:30000], "chunk": c.text} for c in chunks],
        config={"max_concurrency": 8},
    )
    for c, text in zip(chunks, contexts):
        c.extra["context"] = text
    return chunks


def ingest_document(doc: RawDocument, index: VectorIndex, settings: Settings) -> int:
    chunks = enrich_chunks(doc, chunk_document(doc, settings), settings)
    if not chunks:  # document emptied: its previous chunks must not stay searchable
        index.delete_source(doc.tenant, doc.source)
        return 0
    n = index.upsert(chunks)
    log.info("ingested source=%s tenant=%s chunks=%d", doc.source, doc.tenant, n)
    return n


def ingest_path(path: Path, index: VectorIndex, settings: Settings, tenant: str = "default") -> int:
    if path.is_dir():
        return sum(ingest_document(d, index, settings) for d in iter_corpus(path, tenant))
    return ingest_document(load_file(path, tenant), index, settings)
