"""Load test: replay the golden questions as traffic.

    pip install locust
    locust -f benchmarks/locustfile.py --host http://localhost:8000

Read: p50/p95/p99 latency, error rate, requests/s at saturation, and *cost per request* from
LangSmith. Run it against the real provider config, otherwise the numbers are meaningless.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    json.loads(line)["question"]
    for line in (Path(__file__).parent.parent / "evals" / "dataset.jsonl").read_text().splitlines()
    if line.strip()
]


class RagUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask(self):
        self.client.post(
            "/v1/query",
            json={"question": random.choice(QUESTIONS)},
            headers={"X-API-Key": "dev-key"},
        )
