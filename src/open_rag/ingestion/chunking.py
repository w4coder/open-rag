"""Structure-aware, parent-child chunking.

Strategy (see docs/adr/0002-chunking.md):
1. Split on markdown headings -> sections keep their semantic boundary + heading path.
2. Split each section into *parent* blocks (~2400 chars) then *child* chunks (~800).
3. Index/search the small children (precise embeddings); return the parent text to the
   LLM (enough context). This is "small-to-big" / parent-document retrieval.

WHY ids are deterministic: re-ingesting an unchanged document yields the same ids, so
upserts are idempotent (safe retries, no duplicates).
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from open_rag.config import Settings
from open_rag.ingestion.loaders import RawDocument

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


@dataclass
class Chunk:
    id: str
    text: str  # child text, used for search
    parent_text: str  # larger context, used for generation
    source: str
    tenant: str
    section: str = ""
    doc_hash: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text actually embedded/indexed: optional LLM context + section path + chunk."""
        prefix = self.extra.get("context", "")
        head = f"{self.section}\n" if self.section else ""
        return f"{prefix}\n{head}{self.text}".strip()

    def payload(self) -> dict:
        return {
            "chunk_id": self.id,
            "text": self.text,
            "parent_text": self.parent_text,
            "source": self.source,
            "tenant": self.tenant,
            "section": self.section,
            "doc_hash": self.doc_hash,
            **self.extra,
        }


def make_chunk_id(tenant: str, source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{tenant}|{source}|{index}|{text}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))  # Qdrant accepts UUID point ids


def chunk_document(doc: RawDocument, settings: Settings) -> list[Chunk]:
    md_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size, chunk_overlap=0
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    chunks: list[Chunk] = []
    index = 0
    for section in md_splitter.split_text(doc.text):
        path = " > ".join(section.metadata[k] for _, k in _HEADERS if k in section.metadata)
        for parent in parent_splitter.split_text(section.page_content):
            for child in child_splitter.split_text(parent):
                if len(child.strip()) < 20:  # drop noise (page numbers, empty headings)
                    continue
                chunks.append(
                    Chunk(
                        id=make_chunk_id(doc.tenant, doc.source, index, child),
                        text=child,
                        parent_text=parent,
                        source=doc.source,
                        tenant=doc.tenant,
                        section=path,
                        doc_hash=doc.content_hash,
                    )
                )
                index += 1
    return chunks
