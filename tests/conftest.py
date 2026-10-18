"""Fixtures. By default tests use embedded in-memory Qdrant; set RAG_TEST_QDRANT_URL
(e.g. http://localhost:6333) to run the same suite against a real Qdrant server.
Each test gets its own alias/collections, so runs never interfere."""
import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from open_rag.config import Settings
from open_rag.indexing.store import VectorIndex
from open_rag.ingestion.pipeline import ingest_path
from open_rag.providers import get_embeddings
from open_rag.service import RagService

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
TEST_QDRANT_URL = os.environ.get("RAG_TEST_QDRANT_URL", "")


def new_client() -> QdrantClient:
    return QdrantClient(url=TEST_QDRANT_URL) if TEST_QDRANT_URL else QdrantClient(":memory:")


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", embedding_provider="fake", _env_file=None,
                    collection_alias=f"t{uuid.uuid4().hex[:10]}")


@pytest.fixture
def service(settings) -> RagService:
    index = VectorIndex(settings, get_embeddings(settings), client=new_client())
    svc = RagService(settings, index=index)
    ingest_path(CORPUS, index, settings, "tenant-a")
    return svc


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """Environment for the FastAPI app: embedded Qdrant on disk, or the test server.

    Hermetic: run from tmp_path so the developer's .env (real providers, keys) is never read,
    and pin the offline providers explicitly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "fake")
    if TEST_QDRANT_URL:
        monkeypatch.setenv("RAG_QDRANT_URL", TEST_QDRANT_URL)
    monkeypatch.setenv("RAG_QDRANT_PATH", str(tmp_path / "q"))
    monkeypatch.setenv("RAG_COLLECTION_ALIAS", f"t{uuid.uuid4().hex[:10]}")
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "up"))
    return tmp_path
