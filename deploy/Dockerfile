# --- build stage: resolve pinned deps into a venv (reproducible: requirements.lock) ---
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY requirements.lock pyproject.toml ./
RUN python -m venv /venv && /venv/bin/pip install -r requirements.lock
COPY src ./src
RUN /venv/bin/pip install --no-deps .

# --- runtime stage: no compiler, no source tree, non-root, read-only friendly ---
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH=/venv/bin:$PATH RAG_LOG_JSON=true
COPY --from=build /venv /venv
RUN useradd --uid 10001 --create-home app && mkdir -p /app/data/uploads && chown -R 10001 /app
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=2)"
# One worker per container: scale by replicas (k8s HPA), not by processes.
# Graceful shutdown: in-flight requests get 20 s after SIGTERM (k8s terminationGracePeriod 30 s).
CMD ["uvicorn", "open_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20", "--no-server-header"]
