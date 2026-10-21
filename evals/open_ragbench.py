"""Open RAG Bench (vectara/open_ragbench) evaluation - text-only subset, real RAG pipeline.

    python -m evals.open_ragbench                          # 100 text queries + 200 distractor papers
    python -m evals.open_ragbench --queries 300 --negatives 600 --reranker cross-encoder
    python -m evals.open_ragbench --no-judge               # retrieval + e2e, no LLM judge

WHY this benchmark: 1000 arXiv papers (400 answer-bearing, 600 *hard negatives* from the same
domains) and 3045 LLM-generated questions with reference answers and (doc, section) qrels.
The demo corpus (13 chunks) cannot show hybrid/rerank gains; this one can.

Choices worth knowing:
* Text-only queries (`source == "text"`): our loaders ignore images/tables, so multimodal
  queries would measure the parser, not the RAG. Multimodal comes later.
* We ingest the dataset's *pre-parsed* markdown sections (corpus/*.json), not the PDFs: this
  isolates retrieval/generation quality from PDF parsing. Each section is one document
  (`source = "<doc_id>#<section_id>"`) so section-level qrels are checkable; the paper title is
  prepended as an h1, so it lands in every chunk's heading path.
* Gold = qrels (doc/section ids), not text snippets: abstractive answers have no verbatim span.
* Answer quality = LLM judge (reference-guided, 1-5): lexical overlap fails on free-form answers.
  Use a judge model >= the generator, ideally a different one (self-preference bias).
* The index is persisted and keyed by (embedding, chunking) and the subset by (queries,
  negatives, seed): re-runs that only change retrieval/generation settings re-embed nothing.

Data is CC-BY-NC-4.0 and cached under data/open_ragbench/ (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import threading
import time
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from qdrant_client import models

from open_rag.config import Settings
from open_rag.indexing.store import Hit
from open_rag.ingestion.chunking import chunk_document
from open_rag.ingestion.loaders import RawDocument
from open_rag.ingestion.pipeline import enrich_chunks
from open_rag.models import QueryRequest
from open_rag.service import RagService

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "open_ragbench"
OUT = ROOT / "evals" / "results"
HF = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # markdown image refs: no pixels in text-only mode


# ---- dataset ------------------------------------------------------------------------------
def fetch(rel: str) -> Path:
    dst = CACHE / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(f"{HF}/{rel}", tmp)
        tmp.replace(dst)
    return dst


def load_json(rel: str):
    return json.loads(fetch(rel).read_text(encoding="utf-8"))


def select_subset(n_queries: int, n_negatives: int, seed: int) -> tuple[list[dict], list[str]]:
    """Seeded sample of text-only queries + their gold papers + distractor papers.

    Distractors come from the 600 hard negatives first (never the answer to any query)."""
    queries, qrels, answers = load_json("queries.json"), load_json("qrels.json"), load_json("answers.json")
    all_docs = set(load_json("pdf_urls.json"))
    rng = random.Random(seed)
    text_ids = sorted(k for k, v in queries.items() if v["source"] == "text")
    picked = rng.sample(text_ids, min(n_queries, len(text_ids)))
    items = [{"id": q, "question": queries[q]["query"], "type": queries[q]["type"],
              "reference": answers[q], "doc": qrels[q]["doc_id"],
              "section": f"{qrels[q]['doc_id']}#{qrels[q]['section_id']}"} for q in picked]
    gold = {it["doc"] for it in items}
    hard = sorted(all_docs - {r["doc_id"] for r in qrels.values()})
    other = sorted(all_docs - gold - set(hard))
    pool = rng.sample(hard, len(hard)) + rng.sample(other, len(other))
    return items, sorted(gold) + sorted(pool[:n_negatives])


def unanswerable_items(items: list[dict], doc_ids: list[str], n_holdout: int, seed: int,
                       max_absent: int = 100
                       ) -> tuple[list[dict], list[str]]:
    """Questions the corpus cannot answer: the system must refuse them.

    * natural: questions (any modality) whose paper is not indexed at all;
    * holdout: the text questions of `n_holdout` gold papers, asked against a copy of the index
      without those papers. Same questions, same distractors: only the answer is missing.
    Returns the items (answerable=False) and the held-out paper ids."""
    queries, qrels = load_json("queries.json"), load_json("qrels.json")
    indexed = set(doc_ids)
    absent = [q for q in sorted(queries) if qrels[q]["doc_id"] not in indexed]
    absent = random.Random(seed + 2).sample(absent, min(max_absent, len(absent)))  # a subset leaves thousands
    out = [{"id": f"{q}:absent", "question": queries[q]["query"], "type": queries[q]["type"],
            "reference": "", "doc": qrels[q]["doc_id"], "section": "", "answerable": False,
            "corpus": "main"} for q in sorted(absent)]
    gold = sorted({it["doc"] for it in items})
    held = sorted(random.Random(seed + 1).sample(gold, min(n_holdout, len(gold))))
    out += [{**it, "id": f"{it['id']}:holdout", "reference": "", "answerable": False,
             "corpus": "holdout"} for it in items if it["doc"] in set(held)]
    return out, held


def paper_documents(doc_id: str, tenant: str) -> list[RawDocument]:
    paper = load_json(f"corpus/{doc_id}.json")
    title = " ".join(paper["title"].split())
    docs = []
    for s in paper["sections"]:
        text = _IMAGE.sub("", s["text"]).strip()
        if text:
            docs.append(RawDocument(source=f"{doc_id}#{s['section_id']}", tenant=tenant,
                                    text=f"# {title}\n\n{text}"))
    return docs


# ---- indexing -----------------------------------------------------------------------------
def index_alias(s: Settings) -> str:
    """Anything that changes the stored vectors must change the collection."""
    key = [s.embedding_provider, s.embedding_model, s.embedding_dim, s.chunk_size,
           s.chunk_overlap, s.parent_chunk_size,
           s.contextual_enrichment and s.llm_model]  # the LLM only matters if it writes context
    return "orb_" + hashlib.sha1(json.dumps(key).encode()).hexdigest()[:10]


def ensure_indexed(svc: RagService, doc_ids: list[str], tenant: str) -> None:
    with ThreadPoolExecutor(8) as pool:  # parallel download, cached on disk
        papers = list(pool.map(lambda d: paper_documents(d, tenant), doc_ids))
    flt = lambda src: models.Filter(must=[  # noqa: E731
        models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
        models.FieldCondition(key="source", match=models.MatchValue(value=src))])
    # The last section is written last: if it is there, the whole paper is.
    todo = [p for p in papers if p and not svc.index.client.count(
        svc.index.alias, count_filter=flt(p[-1].source), exact=True).count]
    print(f"index {svc.index.alias}/{tenant}: {len(papers) - len(todo)} papers cached, "
          f"{len(todo)} to ingest")
    started = time.perf_counter()
    inner = svc.index.embeddings
    svc.index.embeddings = cache = _Prefetched(inner)

    def prepare(sections: list[RawDocument]) -> list:
        chunks = [c for doc in sections
                  for c in enrich_chunks(doc, chunk_document(doc, svc.settings), svc.settings)]
        cache.prefetch([c.embed_text for c in chunks])  # network-bound: runs in threads
        return chunks

    done = 0
    try:
        with ThreadPoolExecutor(8) as pool:
            for w in range(0, len(todo), 8):
                for chunks in pool.map(prepare, todo[w : w + 8]):
                    # Bulk path, sequential (embedded Qdrant is not thread-safe): one upsert per
                    # paper, replace=False since a paper not yet indexed has no stale points.
                    # The per-document delete is a full scan in embedded Qdrant (quadratic).
                    svc.index.upsert(chunks, replace=False)
                    done += 1
                cache.clear()
                print(f"  ingested {done}/{len(todo)} papers ({time.perf_counter() - started:.0f}s)")
    finally:
        svc.index.embeddings = inner


def ensure_holdout(svc: RagService, tenant: str, holdout: str, held: list[str]) -> None:
    """Copy the tenant's points (vectors included: no re-embedding) minus the held-out papers
    into another tenant. A marker file records a completed copy."""
    marker = CACHE / f"holdout_{svc.index.alias}_{holdout}.done"
    if marker.exists():
        print(f"holdout corpus {holdout}: cached")
        return
    excluded, client, alias = set(held), svc.index.client, svc.index.alias
    flt = models.Filter(must=[models.FieldCondition(key="tenant",
                                                    match=models.MatchValue(value=tenant))])
    offset, n, started = None, 0, time.perf_counter()
    while True:
        pts, offset = client.scroll(alias, scroll_filter=flt, limit=512, offset=offset,
                                    with_payload=True, with_vectors=True)
        keep = [p for p in pts if p.payload["source"].split("#", 1)[0] not in excluded]
        if keep:
            client.upsert(alias, wait=True, points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{holdout}|{p.id}")), vector=p.vector,
                payload={**p.payload, "tenant": holdout}) for p in keep])
            n += len(keep)
        if offset is None:
            break
    marker.write_text(f"{n} points, {len(held)} papers held out\n", encoding="utf-8")
    print(f"holdout corpus {holdout}: copied {n} points ({time.perf_counter() - started:.0f}s)")


class _Prefetched(Embeddings):
    """Embeds several papers concurrently, then serves the vectors to the sequential upsert."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner, self.cache, self.lock = inner, {}, threading.Lock()

    def prefetch(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        vectors = self._embed_with_backoff(missing) if missing else []
        with self.lock:
            self.cache.update(zip(missing, vectors))

    def _embed_with_backoff(self, texts: list[str], attempts: int = 8) -> list[list[float]]:
        """Bulk ingestion hits the provider's tokens-per-minute cap and slow responses on big
        batches: wait them out, don't fail a 20-minute ingestion."""
        transient = ("ratelimit", "timeout", "connection")
        for attempt in range(1, attempts + 1):
            try:
                return self.inner.embed_documents(texts)
            except Exception as e:
                name = type(e).__name__.lower()
                if not any(t in name for t in transient) or attempt == attempts:
                    raise
                time.sleep(min(60, 8 * attempt))
        raise AssertionError("unreachable")

    def clear(self) -> None:
        self.cache.clear()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.prefetch(texts)
        return [self.cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


# ---- retrieval metrics --------------------------------------------------------------------
def _doc(source: str) -> str:
    return source.split("#", 1)[0]


def rank_metrics(hits: list[Hit], it: dict, k: int) -> dict:
    docs = [_doc(h.source) for h in hits]
    first = next((i for i, d in enumerate(docs, 1) if d == it["doc"]), None)
    return {f"doc_hit@{k}": float(first is not None and first <= k),
            "doc_mrr": 1.0 / first if first else 0.0,
            f"section_hit@{k}": float(any(h.source == it["section"] for h in hits[:k])),
            "doc_in_candidates": float(first is not None)}


def eval_retrieval(svc: RagService, items: list[dict], tenant: str, workers: int = 1
                   ) -> tuple[dict, list[dict]]:
    """Ablation on the raw question (no rewrite). k = top_k_rerank = what the LLM sees."""
    k, r = svc.settings.top_k_rerank, svc.retriever
    variants = {"dense_only": lambda q: r.retrieve_dense_only(q, tenant),
                "sparse_only": lambda q: r.retrieve_sparse_only(q, tenant),
                "hybrid_rrf": lambda q: r.retrieve(q, tenant)}
    if svc.settings.reranker != "none":
        variants[f"hybrid+{svc.settings.reranker}"] = lambda q: svc.reranker.rerank(
            q, r.retrieve(q, tenant), k)
    per_item: list[dict] = [{} for _ in items]
    summary = {}
    for name, fn in variants.items():
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda it, fn=fn: rank_metrics(fn(it["question"]), it, k), items))
        print(f"  {name}: done")
        for row, store in zip(rows, per_item):
            store[name] = row
        summary[name] = {m: round(statistics.mean(x[m] for x in rows), 3) for m in rows[0]}
    return summary, per_item


# ---- end to end ---------------------------------------------------------------------------
def run_queries(svc: RagService, items: list[dict], tenants: dict[str, str], workers: int
                ) -> list[dict]:
    """`tenants` maps an item's corpus ("main", "holdout") to the tenant it is asked in."""
    def one(it: dict) -> dict:
        try:
            r = svc.query(QueryRequest(question=it["question"]), tenants[it.get("corpus", "main")])
            return {"answer": r.answer, "status": r.trace.get("status"),
                    "reason": r.trace.get("reason"),
                    "cited": [c.source for c in r.citations],
                    "cited_ids": [c.chunk_id for c in r.citations],
                    "escalated": r.trace.get("escalated", False),
                    "latency_ms": r.trace.get("latency_ms")}
        except Exception as e:  # one failure must not kill a 20-minute run
            return {"answer": "", "status": "error", "cited": [], "error": str(e)[:300]}

    out: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(one, items), 1):
            out.append(res)
            if i % 10 == 0 or i == len(items):
                print(f"  answered {i}/{len(items)}")
    return out


class Verdict(BaseModel):
    reasoning: str = Field(description="One or two sentences comparing candidate and reference.")
    score: int = Field(ge=1, le=5, description="1-5 per the rubric.")


JUDGE_SYSTEM = """You grade answers of a question-answering system against a reference answer.
Judge semantic correctness, not wording. Extra correct detail is fine; contradicting the
reference or inventing facts is not.
5 = covers all key facts of the reference, nothing wrong
4 = essentially correct, a minor fact missing
3 = partially correct: some key facts missing or slightly wrong
2 = mostly wrong or missing the point
1 = wrong, contradicts the reference, or does not answer"""

JUDGE_HUMAN = "Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {answer}"


def judge(items: list[dict], answers: list[dict], model: str, concurrency: int) -> list[dict]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    chain = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)]) \
        | ChatOpenAI(model=model, temperature=0, max_retries=3).with_structured_output(Verdict)
    # Refusals / failures on answerable questions score 1 without an LLM call. Unanswerable
    # questions are not judged: refusing them is the expected behaviour (see summarize_e2e).
    todo = [i for i, a in enumerate(answers)
            if a["status"] == "ok" and items[i].get("answerable", True)]
    verdicts = chain.batch(
        [{"question": items[i]["question"], "reference": items[i]["reference"],
          "answer": answers[i]["answer"]} for i in todo],
        config={"max_concurrency": concurrency}, return_exceptions=True)
    out = [{"score": 1, "reasoning": f"not judged (status={a['status']})"}
           if it.get("answerable", True) else {"score": None, "reasoning": "unanswerable"}
           for it, a in zip(items, answers)]
    for i, v in zip(todo, verdicts):
        out[i] = ({"score": None, "reasoning": f"judge error: {v}"[:300]}
                  if isinstance(v, Exception) else v.model_dump())
    return out


class Grounding(BaseModel):
    verdict: str = Field(description="supported | partly_supported | unsupported")


GROUNDING_SYSTEM = """Is the ANSWER supported by the CONTEXT passages the system cited?
supported: every claim is in the context; partly_supported: some claims are not;
unsupported: the answer relies on facts absent from the context."""


def judge_grounding(svc: RagService, items: list[dict], answers: list[dict], model: str,
                    concurrency: int) -> list[dict | None]:
    """For answers to *unanswerable* questions: a question whose own paper is missing can
    still be answered by another paper (generic questions), so a refusal-only metric
    penalises legitimate answers. What must not happen is an answer the cited passages do
    not support."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    chain = ChatPromptTemplate.from_messages([("system", GROUNDING_SYSTEM), ("human",
        "QUESTION: {question}\nANSWER: {answer}\n\nCONTEXT:\n{context}")]) \
        | ChatOpenAI(model=model, temperature=0, max_retries=3).with_structured_output(Grounding)
    todo = [i for i, (it, a) in enumerate(zip(items, answers))
            if not it.get("answerable", True) and a.get("status") == "ok"]
    inputs = []
    for i in todo:
        pts = svc.index.client.retrieve(svc.index.alias, ids=answers[i]["cited_ids"],
                                        with_payload=True) if answers[i]["cited_ids"] else []
        context = "\n\n".join(p.payload.get("parent_text", p.payload["text"]) for p in pts)
        inputs.append({"question": items[i]["question"], "answer": answers[i]["answer"],
                       "context": context or "(no passage cited)"})
    verdicts = chain.batch(inputs, config={"max_concurrency": concurrency},
                           return_exceptions=True)
    out: list[dict | None] = [None] * len(items)
    for i, v in zip(todo, verdicts):
        out[i] = {"verdict": "error"} if isinstance(v, Exception) else v.model_dump()
    return out


def summarize_e2e(items: list[dict], answers: list[dict], verdicts: list[dict] | None,
                  grounding: list[dict | None] | None = None) -> dict:
    def block(idx: list[int]) -> dict:
        ans = [answers[i] for i in idx]
        answered = [i for i in idx if answers[i]["status"] == "ok"]
        res = {"n": len(idx),
               "answered_rate": round(len(answered) / len(idx), 3),
               "error_rate": round(sum(a["status"] == "error" for a in ans) / len(idx), 3),
               # Of the answered questions: does a citation point into the gold paper?
               "citation_gold_doc": round(statistics.mean(
                   float(any(_doc(s) == items[i]["doc"] for s in answers[i]["cited"]))
                   for i in answered), 3) if answered else 0.0}
        if verdicts is not None:
            scores = [verdicts[i]["score"] for i in idx if verdicts[i]["score"] is not None]
            res["correctness"] = round(statistics.mean(float(s >= 4) for s in scores), 3)
            res["mean_score_1_5"] = round(statistics.mean(scores), 2)
        res["escalation_rate"] = round(sum(bool(a.get("escalated")) for a in ans) / len(idx), 3)
        lat = [a["latency_ms"] for a in ans if a.get("latency_ms")]
        if lat:
            res["p50_latency_ms"] = round(statistics.median(lat))
        return res

    answerable = [i for i, it in enumerate(items) if it.get("answerable", True)]
    by_type: dict[str, list[int]] = defaultdict(list)
    for i in answerable:
        by_type[items[i]["type"]].append(i)
    out = {"all": block(answerable), **{t: block(ix) for t, ix in sorted(by_type.items())}}
    unans = [i for i, it in enumerate(items) if not it.get("answerable", True)]
    if unans:  # the right behaviour is a refusal: any answer here is unsupported by the corpus
        refused = lambda ix: round(sum(answers[i]["status"] != "ok" for i in ix) / len(ix), 3)  # noqa: E731
        unsupported = sum(1 for i in unans if grounding and grounding[i]
                          and grounding[i]["verdict"] != "supported")
        out["unanswerable"] = {"n": len(unans), "refusal_rate": refused(unans),
                               "unsupported_answer_rate": round(unsupported / len(unans), 3)
                               if grounding else None, **{
            f"{c}_n": len(ix) for c in ("main", "holdout")
            if (ix := [i for i in unans if items[i]["corpus"] == c])}}
    return out


# ---- main ---------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100, help="text queries (1914 = all of them)")
    ap.add_argument("--negatives", type=int, default=200, help="distractor papers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reranker", choices=["none", "cross-encoder", "llm"])
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--holdout-papers", type=int, default=0,
                    help="gold papers removed from a copy of the index: their questions become "
                         "unanswerable there (refusal test)")
    ap.add_argument("--absent", type=int, default=100,
                    help="max questions about papers that are not indexed at all (refusal test)")
    ap.add_argument("--tag", help="also save results as results/orb_<tag>.json (+ _details.jsonl)")
    ap.add_argument("--reuse-retrieval", metavar="TAG",
                    help="reuse the retrieval ablation of results/orb_<TAG> (same questions and "
                         "index): for generation-only experiments")
    ap.add_argument("--workers", type=int, help="parallel queries (default: 4 with a Qdrant "
                    "server, 1 with embedded Qdrant which is not thread-safe)")
    args = ap.parse_args()

    base = Settings(cache_enabled=False)  # measure the pipeline, not the cache
    overrides = {k: v for k, v in {"reranker": args.reranker, "chunk_size": args.chunk_size}.items()
                 if v is not None}
    s = base.model_copy(update=overrides)
    s = s.model_copy(update={"collection_alias": index_alias(s),
                             "qdrant_path": str(CACHE / "qdrant")})
    if "fake" in (s.llm_provider, s.embedding_provider):
        print("WARNING: fake provider(s) - this is a plumbing smoke test, not a quality "
              "measure. Set RAG_LLM_PROVIDER=openai and RAG_EMBEDDING_PROVIDER=openai in .env.")
    judge_on = not (args.no_judge or args.retrieval_only)

    items, doc_ids = select_subset(args.queries, args.negatives, args.seed)
    tenant = f"orb-q{args.queries}-n{args.negatives}-s{args.seed}"
    print(f"{len(items)} text queries, {len(doc_ids)} papers "
          f"({len({it['doc'] for it in items})} gold)")
    svc = RagService(s)
    ensure_indexed(svc, doc_ids, tenant)
    workers = args.workers or (4 if s.qdrant_url else 1)
    tenants = {"main": tenant, "holdout": f"{tenant}-h{args.holdout_papers}"}
    unans, held = unanswerable_items(items, doc_ids, args.holdout_papers, args.seed,
                                     args.absent)
    if held:
        ensure_holdout(svc, tenant, tenants["holdout"], held)
    print(f"{len(unans)} unanswerable questions ({len(held)} papers held out)")

    if args.reuse_retrieval:
        # Generation-only change (model, prompt, escalation): the retrieval ablation of an
        # earlier run on the same questions and index is still valid, and costly to redo.
        prev = json.loads((OUT / f"orb_{args.reuse_retrieval}.json").read_text(encoding="utf-8"))
        prev_ranks = {r["id"]: r["retrieval"] for r in map(json.loads, (
            OUT / f"orb_{args.reuse_retrieval}_details.jsonl").read_text(encoding="utf-8").splitlines())}
        retrieval, ranks = prev["retrieval"], [prev_ranks[it["id"]] for it in items]
        print(f"retrieval ablation reused from {args.reuse_retrieval}")
    else:
        print("retrieval ablation...")
        retrieval, ranks = eval_retrieval(svc, items, tenant, workers)
    ranks += [{} for _ in unans]
    items = items + unans
    results: dict = {
        "benchmark": "vectara/open_ragbench (text-only)",
        "config": {"llm_provider": s.llm_provider, "llm_model": s.llm_model,
                   "embedding": f"{s.embedding_provider}:{s.embedding_model}",
                   "escalation_model": s.llm_escalation_model or None,
                   "retrieval_reused_from": args.reuse_retrieval,
                   "reranker": s.reranker, "chunk_size": s.chunk_size,
                   "top_k_dense": s.top_k_dense, "top_k_sparse": s.top_k_sparse,
                   "top_k_rerank": s.top_k_rerank, "judge_model": args.judge_model if judge_on else None,
                   "queries": len(items) - len(unans), "unanswerable": len(unans),
                   "holdout_papers": len(held), "papers": len(doc_ids), "seed": args.seed,
                   "chunks_indexed": svc.index.count(tenant)},
        "retrieval": retrieval,
    }
    answers: list[dict] = [{} for _ in items]
    verdicts = grounding = None
    if not args.retrieval_only:
        print("end-to-end...")
        answers = run_queries(svc, items, tenants, workers)
        if judge_on:
            print(f"judging with {args.judge_model}...")
            verdicts = judge(items, answers, args.judge_model, concurrency=8)
            grounding = judge_grounding(svc, items, answers, args.judge_model, concurrency=8)
        results["end_to_end"] = summarize_e2e(items, answers, verdicts, grounding)

    OUT.mkdir(exist_ok=True)
    details = "".join(  # per-question record: the raw material of error analysis
        json.dumps({**it, **answers[i], "retrieval": ranks[i],
                    "judge": verdicts[i] if verdicts else None,
                    "grounding": grounding[i] if grounding else None},
                   ensure_ascii=False) + "\n"
        for i, it in enumerate(items))
    for name in ["open_ragbench_latest", *([f"orb_{args.tag}"] if args.tag else [])]:
        (OUT / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (OUT / f"{name}_details.jsonl").write_text(details, encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
