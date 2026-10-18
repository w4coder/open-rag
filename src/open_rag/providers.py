"""LLM providers (chat + embeddings) behind factories.

The `fake` providers make the entire system runnable offline: CI, unit tests and the
demo need no API key. They are deterministic, so retrieval tests are reproducible.

Resilience: every real chat model has a timeout + bounded retries, and an optional fallback
model (`RAG_LLM_FALLBACK_MODEL`) used when the primary fails (outage, rate limit, parse error).
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from open_rag.config import Settings

_TOKEN = re.compile(r"[a-zà-ÿ0-9]+")


def _stem(tok: str) -> str:
    """Very light English suffix stripping so 'retained' ~ 'retains' (BM25 + offline demo).
    Production: use a real analyzer (Snowball) or let the search engine do it."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


def token_hash(tok: str) -> int:
    """Stable 63-bit hash (Python's hash() is salted per process)."""
    return int(hashlib.md5(tok.encode()).hexdigest()[:15], 16)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (offline stand-in, NOT semantic).

    Good enough to exercise the pipeline and lexical-overlap retrieval in tests.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, timeout=settings.llm_timeout_s,
                                max_retries=settings.llm_max_retries)
    return HashingEmbeddings(settings.embedding_dim)


def _openai(settings: Settings, model: str):
    from langchain_openai import ChatOpenAI

    # temperature=0: RAG answers should be reproducible and extractive.
    return ChatOpenAI(model=model, temperature=0, timeout=settings.llm_timeout_s,
                      max_retries=settings.llm_max_retries)


def get_chat_model(settings: Settings):
    """Primary chat model, or None when running with the offline `fake` provider."""
    if settings.llm_provider == "openai":
        return _openai(settings, settings.llm_model)
    return None


def get_fallback_model(settings: Settings):
    if settings.llm_provider == "openai" and settings.llm_fallback_model:
        return _openai(settings, settings.llm_fallback_model)
    return None


def with_fallback(primary: Runnable, fallback: Runnable | None) -> Runnable:
    return primary.with_fallbacks([fallback]) if fallback is not None else primary


def structured_llm(settings: Settings, schema) -> Runnable | None:
    """`llm.with_structured_output(schema)` + the same on the fallback model.

    Fallbacks are attached *after* structuring so a parse failure also triggers them.
    """
    llm = get_chat_model(settings)
    if llm is None:
        return None
    fb = get_fallback_model(settings)
    return with_fallback(llm.with_structured_output(schema),
                         fb.with_structured_output(schema) if fb is not None else None)


def text_llm(settings: Settings) -> Runnable | None:
    llm = get_chat_model(settings)
    return None if llm is None else with_fallback(llm, get_fallback_model(settings))
