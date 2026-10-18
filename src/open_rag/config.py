"""Central, typed configuration (12-factor: everything comes from env vars / .env).

WHY: a single validated settings object avoids "magic" os.environ reads scattered
around the code and makes every tunable (chunk size, top-k, ...) an explicit
experiment variable that can be compared in LangSmith.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict


def export_sdk_env(path: str = ".env") -> None:
    """Export the .env keys that third-party SDKs read from os.environ (OPENAI_API_KEY,
    LANGSMITH_*). pydantic-settings only parses RAG_* into Settings and never exports, so
    without this the key in .env is silently ignored. Real env vars always win."""
    for key, value in dotenv_values(path).items():
        if not key.startswith("RAG_") and value and key not in os.environ:
            os.environ[key] = value


export_sdk_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Providers
    llm_provider: Literal["openai", "fake"] = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str = ""  # e.g. "gpt-4.1-mini": used when the primary fails
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    embedding_provider: Literal["openai", "fake"] = "fake"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    # Vector store
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = ".qdrant"
    collection_alias: str = "docs"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    parent_chunk_size: int = 2400
    contextual_enrichment: bool = False  # LLM-generated chunk context (costly, +recall)

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    reranker: Literal["none", "cross-encoder", "llm"] = "none"
    # Threshold on the *reranker* score (cross-encoder logit or LLM 0..10). Ignored when
    # reranker=none: RRF scores are rank-based and carry no absolute relevance meaning.
    min_relevance: float = 0.0
    rrf_k: int = 60

    # Semantic cache (shared across replicas: stored in Qdrant)
    cache_enabled: bool = True
    cache_threshold: float = 0.97
    cache_ttl_s: int = 3600

    # Infra
    redis_url: str = ""
    api_keys: str = "dev-key:tenant-a"
    api_threads: int = 64  # max concurrent sync pipelines per replica
    max_question_chars: int = 2000
    max_history_messages: int = 12
    upload_dir: str = "data/uploads"  # shared RWX volume (EFS/Filestore) in a cluster
    max_upload_mb: int = 25
    rate_limit_per_min: int = 60  # per tenant, enforced cluster-wide when Redis is set
    tenant_daily_tokens: int = 0  # 0 = unlimited; LLM token budget per tenant per day
    log_json: bool = False

    def api_key_map(self) -> dict[str, str]:
        """Parse 'key:tenant,key2:tenant2' into {key: tenant}."""
        out: dict[str, str] = {}
        for pair in filter(None, (p.strip() for p in self.api_keys.split(","))):
            key, _, tenant = pair.partition(":")
            out[key] = tenant or "default"
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
