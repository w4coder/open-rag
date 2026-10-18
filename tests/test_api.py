from pathlib import Path

from fastapi.testclient import TestClient

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_api_auth_query_and_health(api_env):
    from open_rag.config import get_settings

    get_settings.cache_clear()
    from open_rag.api import app

    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 200
        assert c.post("/v1/query", json={"question": "hi"}).status_code == 401
        with open(CORPUS / "storage.md", "rb") as f:
            r = c.post("/v1/documents", files={"file": ("storage.md", f)},
                       headers={"X-API-Key": "dev-key"})
        assert r.status_code == 202
        import time

        for _ in range(50):
            if c.get(f"/v1/jobs/{r.json()['job_id']}", headers={"X-API-Key": "dev-key"}).json()["status"] == "done":
                break
            time.sleep(0.1)
        resp = c.post("/v1/query", json={"question": "What is the maximum size of one object?"},
                      headers={"X-API-Key": "dev-key"})
        assert resp.status_code == 200 and resp.json()["citations"]
        assert "rag_requests_total" in c.get("/metrics").text
    get_settings.cache_clear()
