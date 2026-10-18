"""Asynchronous ingestion: API enqueues, workers consume.

WHY: parsing + embedding thousands of documents takes minutes and is bursty. Doing it in
the request path blocks the API and couples its scaling to ingestion load. With a queue:
* API stays fast and stateless; workers scale independently (KEDA on stream lag),
* at-least-once processing (ingestion is idempotent, so redelivery is safe),
* backpressure is explicit (stream lag / pending count are metrics).

`RedisQueue` uses **Redis Streams + a consumer group** (not LPUSH/BRPOP, which loses the job
if the worker dies mid-processing):
* a message stays *pending* until the worker XACKs it after success or after scheduling a
  retry; a worker that crashes leaves it pending and another worker reclaims it with
  XAUTOCLAIM once it has been idle longer than the visibility timeout;
* every delivery counts as an attempt (stored in the job hash), so a "poison" document that
  crashes workers ends in the dead-letter stream instead of looping forever;
* retries are scheduled in a sorted set (due time = exponential backoff): the worker never
  sleeps on a failed job and keeps processing others.

`InlineQueue` (default, no Redis) runs jobs in a thread pool - same interface, so dev/tests
need no infrastructure.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from open_rag.config import Settings, get_settings
from open_rag.indexing.store import VectorIndex
from open_rag.ingestion.pipeline import ingest_path
from open_rag.observability import INGEST
from open_rag.providers import get_embeddings

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
VISIBILITY_TIMEOUT_MS = 10 * 60 * 1000  # a job idle this long is considered abandoned
STREAM, GROUP = "rag:ingest", "rag:workers"
DLQ, DELAYED = "rag:ingest:dlq", "rag:ingest:delayed"
_PUBLIC = ("status", "chunks", "error", "attempts")


class JobQueue(Protocol):
    def enqueue(self, path: str, tenant: str) -> str: ...
    def status(self, job_id: str, tenant: str) -> dict | None: ...


def run_job(job: dict, index: VectorIndex, settings: Settings) -> int:
    return ingest_path(Path(job["path"]), index, settings, job["tenant"])


class InlineQueue:
    def __init__(self, index: VectorIndex, settings: Settings) -> None:
        self.index, self.settings = index, settings
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._jobs: dict[str, dict] = {}

    def enqueue(self, path: str, tenant: str) -> str:
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {"status": "queued", "tenant": tenant}

        def _run() -> None:
            try:
                n = run_job({"path": path, "tenant": tenant}, self.index, self.settings)
                self._jobs[job_id] = {"status": "done", "chunks": n, "tenant": tenant}
                INGEST.labels(status="done").inc()
            except Exception as e:
                log.exception("job failed")
                self._jobs[job_id] = {"status": "failed", "error": str(e), "tenant": tenant}
                INGEST.labels(status="failed").inc()

        self._pool.submit(_run)
        return job_id

    def status(self, job_id: str, tenant: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None or job["tenant"] != tenant:  # never reveal other tenants' jobs
            return None
        return {k: v for k, v in job.items() if k in _PUBLIC}


class RedisQueue:
    def __init__(self, url: str = "", client=None,
                 visibility_timeout_ms: int = VISIBILITY_TIMEOUT_MS) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
        self.r, self.visibility_timeout_ms = client, visibility_timeout_ms
        try:
            self.r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as e:  # BUSYGROUP: the group already exists
            if "BUSYGROUP" not in str(e):
                raise

    @staticmethod
    def _key(job_id: str) -> str:
        return f"rag:job:{job_id}"

    # ---- producer side -----------------------------------------------------------------
    def enqueue(self, path: str, tenant: str) -> str:
        job_id = uuid.uuid4().hex
        self.r.hset(self._key(job_id), mapping={"status": "queued", "tenant": tenant,
                                                "path": path, "attempts": 0})
        self.r.xadd(STREAM, {"job_id": job_id})
        return job_id

    def status(self, job_id: str, tenant: str) -> dict | None:
        job = self.r.hgetall(self._key(job_id))
        if not job or job.get("tenant") != tenant:
            return None
        return {k: v for k, v in job.items() if k in _PUBLIC}

    # ---- consumer side -----------------------------------------------------------------
    def _promote_due_retries(self) -> None:
        for job_id in self.r.zrangebyscore(DELAYED, 0, time.time(), start=0, num=100):
            if self.r.zrem(DELAYED, job_id):  # only one worker wins the promotion
                self.r.hset(self._key(job_id), "status", "queued")
                self.r.xadd(STREAM, {"job_id": job_id})

    def _next_messages(self, consumer: str, block_ms: int) -> list[tuple[str, dict]]:
        # 1) jobs abandoned by a dead/stuck worker, 2) new jobs.
        claimed = self.r.xautoclaim(STREAM, GROUP, consumer, self.visibility_timeout_ms,
                                    start_id="0-0", count=1)[1]
        if claimed:
            return claimed
        res = self.r.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=block_ms)
        return res[0][1] if res else []

    def _ack(self, msg_id: str) -> None:
        self.r.xack(STREAM, GROUP, msg_id)
        self.r.xdel(STREAM, msg_id)

    def _process(self, msg_id: str, fields: dict, index: VectorIndex, settings: Settings,
                 consumer: str) -> None:
        job_id, key = fields["job_id"], self._key(fields["job_id"])
        job = self.r.hgetall(key)
        if not job:
            self._ack(msg_id)
            return
        attempt = self.r.hincrby(key, "attempts", 1)  # counts crashed deliveries too
        if attempt > MAX_ATTEMPTS:
            self._dead(job_id, job.get("error", "worker crashed repeatedly"), msg_id)
            return
        self.r.hset(key, mapping={"status": "running", "worker": consumer})
        try:
            n = run_job(job, index, settings)
        except Exception as e:
            log.exception("job %s failed (attempt %d)", job_id, attempt)
            if attempt >= MAX_ATTEMPTS:
                self._dead(job_id, str(e), msg_id)
            else:
                self.r.hset(key, mapping={"status": "retrying", "error": str(e)})
                self.r.zadd(DELAYED, {job_id: time.time() + 2 ** attempt})
                self._ack(msg_id)
                INGEST.labels(status="retry").inc()
            return
        self.r.hset(key, mapping={"status": "done", "chunks": n})
        self.r.hdel(key, "error")
        self._ack(msg_id)
        INGEST.labels(status="done").inc()

    def _dead(self, job_id: str, error: str, msg_id: str) -> None:
        self.r.xadd(DLQ, {"job_id": job_id, "error": error[:1000]})
        self.r.hset(self._key(job_id), mapping={"status": "dead", "error": error[:1000]})
        self._ack(msg_id)
        INGEST.labels(status="dead").inc()

    def work_once(self, index: VectorIndex, settings: Settings, consumer: str,
                  block_ms: int = 5000) -> int:
        self._promote_due_retries()
        msgs = self._next_messages(consumer, block_ms)
        for msg_id, fields in msgs:
            self._process(msg_id, fields, index, settings, consumer)
        return len(msgs)

    def work_forever(self, index: VectorIndex, settings: Settings,
                     stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        consumer = f"{socket.gethostname()}-{os.getpid()}"
        log.info("worker %s started", consumer)
        while not stop.is_set():  # graceful: finishes the current job, then exits
            self.work_once(index, settings, consumer)
        log.info("worker %s stopped", consumer)


def get_queue(settings: Settings, index: VectorIndex) -> JobQueue:
    return RedisQueue(settings.redis_url) if settings.redis_url else InlineQueue(index, settings)


def worker_main() -> None:
    """Entry point: `python -m open_rag.queue` (one process per worker replica)."""
    from prometheus_client import start_http_server

    from open_rag.observability import configure_logging

    settings = get_settings()
    configure_logging(settings.log_json)
    if not settings.redis_url:
        raise SystemExit("RAG_REDIS_URL is required to run a standalone worker")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    start_http_server(int(os.environ.get("RAG_WORKER_METRICS_PORT", "9100")))
    index = VectorIndex(settings, get_embeddings(settings))
    RedisQueue(settings.redis_url).work_forever(index, settings, stop)


if __name__ == "__main__":
    worker_main()
