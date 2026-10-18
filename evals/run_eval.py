"""Offline-capable evaluation harness.

    python -m evals.run_eval                 # retrieval ablation + end-to-end, local
    python -m evals.run_eval --langsmith     # also push dataset + experiment to LangSmith

Experiments compare *variants* on the same dataset - the only honest way to decide whether
a change (hybrid, reranker, chunk size, new embedding model) is an improvement.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from qdrant_client import QdrantClient

from evals.metrics import answer_correct, hit_at_k, reciprocal_rank
from open_rag.config import Settings
from open_rag.ingestion.pipeline import ingest_path
from open_rag.models import QueryRequest
from open_rag.service import RagService

ROOT = Path(__file__).resolve().parent.parent
TENANT = "tenant-a"
K = 5
JUDGE_MODEL = "gpt-4.1-mini"


def load_dataset() -> list[dict]:
    lines = (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_service(**overrides) -> RagService:
    settings = Settings(**{"cache_enabled": False, **overrides})  # measure the pipeline, not the cache
    from open_rag.indexing.store import VectorIndex
    from open_rag.providers import get_embeddings

    index = VectorIndex(settings, get_embeddings(settings), client=QdrantClient(":memory:"))
    svc = RagService(settings, index=index)
    ingest_path(ROOT / "data" / "corpus", svc.index, settings, TENANT)
    return svc


def eval_retrieval(svc: RagService, data: list[dict], mode: str) -> dict:
    hits, rrs = [], []
    for ex in (e for e in data if e["answerable"]):
        if mode == "dense":
            res = svc.retriever.retrieve_dense_only(ex["question"], TENANT)
        elif mode == "sparse":
            res = svc.retriever.retrieve_sparse_only(ex["question"], TENANT)
        else:
            res = svc.retriever.retrieve(ex["question"], TENANT)
        texts = [h.text for h in res]
        hits.append(hit_at_k(texts, ex["gold"], K))
        rrs.append(reciprocal_rank(texts, ex["gold"]))
    return {f"hit@{K}": round(statistics.mean(hits), 3), "mrr": round(statistics.mean(rrs), 3)}


def eval_end_to_end(svc: RagService, data: list[dict]) -> dict:
    refusal_ok, cite_ok, graded, answers = [], [], [], []
    for ex in data:
        resp = svc.query(QueryRequest(question=ex["question"]), TENANT)
        # A refusal is a pipeline outcome, not a phrase: real models word it freely.
        refused = resp.trace.get("status") != "ok"
        if ex["answerable"]:
            graded.append(ex)
            answers.append({"answer": resp.answer, "status": resp.trace.get("status")})
            cite_ok.append(float(bool(resp.citations) and resp.grounded))
        else:
            refusal_ok.append(float(refused))
    if svc.settings.llm_provider == "fake":
        # The offline extractive generator copies the source: a lexical check is exact here.
        correct = [answer_correct(a["answer"], ex["gold"]) for ex, a in zip(graded, answers)]
    else:
        # Real models paraphrase ("rotated every 90 days" for "expires after 90 days"): use
        # the same reference-guided LLM judge as the Open RAG Bench harness.
        from evals.open_ragbench import judge

        verdicts = judge([{**ex, "reference": ex["gold"]} for ex in graded], answers,
                         JUDGE_MODEL, concurrency=4)
        correct = [float((v["score"] or 0) >= 4) for v in verdicts]
    return {
        "answer_correctness": round(statistics.mean(correct), 3),
        "citation_rate": round(statistics.mean(cite_ok), 3),
        "refusal_accuracy": round(statistics.mean(refusal_ok), 3),
    }


def run_langsmith(svc: RagService, data: list[dict], prefix: str) -> None:
    from langsmith import Client, evaluate

    client = Client()
    name = "open-rag-golden"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="open-rag golden Q/A set")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": e["question"]} for e in data],
            outputs=[{"gold": e["gold"], "answerable": e["answerable"]} for e in data],
        )

    def target(inputs: dict) -> dict:
        r = svc.query(QueryRequest(question=inputs["question"]), TENANT)
        return {"answer": r.answer, "grounded": r.grounded, "refused": r.trace.get("status") != "ok",
                "citations": [c.chunk_id for c in r.citations]}

    def correctness(outputs: dict, reference_outputs: dict) -> dict:
        if not reference_outputs["answerable"]:
            return {"key": "correct_or_refused",
                    "score": float(outputs["refused"])}
        return {"key": "correct_or_refused",
                "score": answer_correct(outputs["answer"], reference_outputs["gold"])}

    def has_citations(outputs: dict, reference_outputs: dict) -> dict:
        need = reference_outputs["answerable"]
        return {"key": "citation_valid", "score": float(bool(outputs["citations"]) == need)}

    evaluate(
        target, data=name, evaluators=[correctness, has_citations], experiment_prefix=prefix,
        metadata={"chunk_size": svc.settings.chunk_size, "reranker": svc.settings.reranker,
                  "embedding": svc.settings.embedding_provider},
        max_concurrency=4,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langsmith", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=800)
    args = ap.parse_args()

    data = load_dataset()
    svc = build_service(chunk_size=args.chunk_size)
    results = {
        "config": {"llm_provider": svc.settings.llm_provider,
                   "embedding_provider": svc.settings.embedding_provider,
                   "reranker": svc.settings.reranker, "chunk_size": svc.settings.chunk_size},
        "dense_only": eval_retrieval(svc, data, "dense"),
        "sparse_only": eval_retrieval(svc, data, "sparse"),
        "hybrid_rrf": eval_retrieval(svc, data, "hybrid"),
        "end_to_end": eval_end_to_end(svc, data),
    }
    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if args.langsmith:
        run_langsmith(svc, data, prefix=f"chunk{args.chunk_size}-{svc.settings.reranker}")


if __name__ == "__main__":
    main()
