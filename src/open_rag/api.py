"""HTTP API: auth -> tenant, rate limit, query, streaming, ingestion, health, metrics."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from open_rag.config import get_settings
from open_rag.indexing.store import LegacySchemaError
from open_rag.ingestion.loaders import SUPPORTED_EXTENSIONS
from open_rag.models import IngestResponse, QueryRequest, QueryResponse
from open_rag.observability import configure_logging, request_id
from open_rag.queue import get_queue
from open_rag.service import BudgetExceeded, RagService, UpstreamError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_json)
    # The pipeline is sync (LLM SDK + Qdrant client): it runs in the thread pool, whose size
    # is the per-replica concurrency cap. Scale out with replicas (HPA) beyond that.
    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.api_threads
    app.state.service = RagService(settings)
    app.state.queue = get_queue(settings, app.state.service.index)
    app.state.upload_dir = Path(settings.upload_dir)
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="open-rag", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def correlate(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(UpstreamError)
async def _upstream(_: Request, exc: UpstreamError):
    return JSONResponse({"detail": "upstream dependency unavailable, retry later"}, status_code=503,
                        headers={"Retry-After": "5"})


@app.exception_handler(LegacySchemaError)
async def _legacy(_: Request, exc: LegacySchemaError):
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(BudgetExceeded)
async def _budget(_: Request, exc: BudgetExceeded):
    return JSONResponse({"detail": str(exc)}, status_code=429)


def tenant_from_key(request: Request, x_api_key: str = Header(default="")) -> str:
    tenant = get_settings().api_key_map().get(x_api_key)
    if tenant is None:
        raise HTTPException(401, "Invalid API key")
    if not request.app.state.service.limits.allow_request(tenant):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
    return tenant


@app.post("/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, tenant: str = Depends(tenant_from_key)):
    return await run_in_threadpool(request.app.state.service.query, req, tenant)


@app.post("/v1/query/stream")
async def query_stream(req: QueryRequest, request: Request,
                       tenant: str = Depends(tenant_from_key)):
    """Server-Sent Events: one `step` per pipeline node (progress), then `answer`,
    `citations` and `done` (with the trace). Same cache/budget/metrics path as /v1/query."""
    service: RagService = request.app.state.service
    rid = request_id.get()

    def events():
        request_id.set(rid)  # generator runs in a worker thread: restore correlation id
        try:
            for kind, value in service.stream(req, tenant):
                if kind == "step":
                    yield f"event: step\ndata: {json.dumps({'node': value})}\n\n"
                else:
                    resp: QueryResponse = value
                    yield f"event: answer\ndata: {json.dumps(resp.answer)}\n\n"
                    cites = [c.model_dump() for c in resp.citations]
                    yield f"event: citations\ndata: {json.dumps(cites)}\n\n"
                    yield f"event: done\ndata: {json.dumps({'cached': resp.cached, **resp.trace})}\n\n"
        except (UpstreamError, BudgetExceeded, LegacySchemaError) as e:
            yield f"event: error\ndata: {json.dumps({'detail': type(e).__name__})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/documents", response_model=IngestResponse, status_code=202)
async def upload(request: Request, file: UploadFile = File(...),
                 tenant: str = Depends(tenant_from_key)):
    name = Path(file.filename or "").name  # basename only: prevents path traversal
    suffix = Path(name).suffix.lower()
    if not name or name.startswith(".") or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file {name!r}")
    limit = get_settings().max_upload_mb * 1024 * 1024
    dest_dir = request.app.state.upload_dir / tenant
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".{uuid.uuid4().hex}.part"
    size = 0
    try:
        with tmp.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MB")
                f.write(chunk)
        os.replace(tmp, dest_dir / name)  # atomic: workers never read a partial file
    finally:
        tmp.unlink(missing_ok=True)
    job_id = request.app.state.queue.enqueue(str(dest_dir / name), tenant)
    return IngestResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str, request: Request, tenant: str = Depends(tenant_from_key)):
    status = request.app.state.queue.status(job_id, tenant)
    if status is None:  # unknown OR owned by another tenant: same answer, no oracle
        raise HTTPException(404, "Unknown job")
    return status


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}  # liveness: process is up


@app.get("/readyz")
async def readyz(request: Request):
    service: RagService = request.app.state.service
    checks = {"qdrant": await run_in_threadpool(service.index.healthy)}
    r = getattr(request.app.state.queue, "r", None)
    if r is not None:
        try:
            checks["redis"] = bool(await run_in_threadpool(r.ping))
        except Exception:
            checks["redis"] = False
    if not all(checks.values()):
        raise HTTPException(503, f"dependencies unavailable: {checks}")
    return {"status": "ready", **checks}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
