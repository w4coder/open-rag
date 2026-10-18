"""Metrics, structured logs and request correlation.

* Prometheus: request status, end-to-end + per-node latency, cache hits, LLM tokens per tenant
  and model (cost dashboards / alerts), ingestion outcomes.
* Logs: optional JSON lines (`RAG_LOG_JSON=true`) carrying the request id, so a log line, a
  LangSmith trace (metadata.request_id) and an HTTP response (X-Request-ID) can be joined.
"""
from __future__ import annotations

import contextvars
import functools
import json
import logging
import time

from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_requests_total", "RAG requests", ["status"])
LATENCY = Histogram("rag_latency_seconds", "End-to-end RAG latency",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
NODE_LATENCY = Histogram("rag_node_latency_seconds", "Latency per pipeline node", ["node"],
                         buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
CACHE = Counter("rag_cache_total", "Semantic cache lookups", ["result"])
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["tenant", "model", "kind"])
INGEST = Counter("rag_ingest_jobs_total", "Ingestion job outcomes", ["status"])

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def timed_node(name: str, fn):
    """Wrap a graph node so its latency lands in NODE_LATENCY."""
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        try:
            return fn(state)
        finally:
            NODE_LATENCY.labels(node=name).observe(time.perf_counter() - start)
    return wrapper


def record_usage(tenant: str, usage: dict) -> int:
    """usage = UsageMetadataCallbackHandler.usage_metadata ({model: {input_tokens, ...}})."""
    total = 0
    for model, u in usage.items():
        for kind in ("input_tokens", "output_tokens"):
            n = int(u.get(kind, 0) or 0)
            TOKENS.labels(tenant=tenant, model=model, kind=kind).inc(n)
            total += n
    return total


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
               "msg": record.getMessage(), "request_id": request_id.get()}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def configure_logging(json_logs: bool) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(_JsonFormatter() if json_logs else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
