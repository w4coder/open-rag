r"""The online RAG pipeline as an explicit LangGraph state machine.

    guard -> rewrite -> retrieve -> rerank -> (enough evidence?) -> generate -> END
      |                                              |
      +--> blocked (END)                             +--> no_evidence (END)

WHY a graph (and not a plain function) for a *linear* RAG: every node becomes a traced,
independently testable step with typed state; branches (blocked / no evidence / ungrounded)
are explicit instead of buried in if-statements; and it is the natural base for the
agentic project (same state + nodes, add loops).
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from open_rag.config import Settings
from open_rag.generation.chains import REFUSAL, build_answerer, build_rewriter
from open_rag.generation.guardrails import (
    GuardrailError,
    check_grounding,
    check_history,
    check_input,
)
from open_rag.indexing.store import Hit
from open_rag.models import Answer, Citation
from open_rag.observability import timed_node
from open_rag.retrieval.hybrid import HybridRetriever
from open_rag.retrieval.rerank import Reranker


class RagState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    tenant: str
    search_query: str
    query_vector: list[float] | None  # embedding of `question`, computed once by the service
    candidates: list[Hit]
    hits: list[Hit]
    answer: str
    citations: list[Citation]
    grounded: bool
    status: str  # ok | blocked | no_evidence | ungrounded
    reason: str


def _history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])


def build_graph(settings: Settings, retriever: HybridRetriever, reranker: Reranker):
    rewriter = build_rewriter(settings)
    answerer = build_answerer(settings)

    def guard(state: RagState) -> RagState:
        try:
            return {"question": check_input(state["question"], settings.max_question_chars),
                    "history": check_history(state.get("history", []),
                                             settings.max_history_messages)}
        except GuardrailError as e:
            return {"status": "blocked", "reason": e.code, "answer": str(e), "grounded": True,
                    "citations": []}

    def rewrite(state: RagState) -> RagState:
        if not state.get("history"):  # first turn: nothing to resolve, save an LLM call
            return {"search_query": state["question"]}
        query = rewriter.invoke(
            {"question": state["question"], "history": _history_text(state["history"])}
        )
        return {"search_query": query, "query_vector": None}  # vector was for the raw question

    def retrieve(state: RagState) -> RagState:
        return {"candidates": retriever.retrieve(state["search_query"], state["tenant"],
                                                 query_vector=state.get("query_vector"))}

    def rerank(state: RagState) -> RagState:
        top = reranker.rerank(state["search_query"], state["candidates"], settings.top_k_rerank)
        if settings.reranker == "none":  # RRF scores are rank-based: no absolute threshold
            return {"hits": top}
        return {"hits": [h for h in top if h.score >= settings.min_relevance]}

    def generate(state: RagState) -> RagState:
        ans: Answer = answerer.invoke({"question": state["question"], "hits": state["hits"]})
        by_id = {h.chunk_id: h for h in state["hits"]}
        report = check_grounding(ans, set(by_id))
        if not report.grounded:
            return {"status": "ungrounded", "reason": report.reason, "answer": REFUSAL,
                    "grounded": False, "citations": []}
        if ans.insufficient_context:
            return {"status": "no_evidence", "answer": ans.answer or REFUSAL, "grounded": True,
                    "citations": []}
        cites = [
            Citation(chunk_id=i, source=by_id[i].source, section=by_id[i].section,
                     snippet=by_id[i].text[:240])
            for i in report.valid_ids
        ]
        return {"status": "ok", "answer": ans.answer, "grounded": True, "citations": cites}

    def no_evidence(state: RagState) -> RagState:
        return {"status": "no_evidence", "answer": REFUSAL, "grounded": True, "citations": []}

    def after_guard(state: RagState) -> str:
        return "end" if state.get("status") == "blocked" else "rewrite"

    def after_rerank(state: RagState) -> str:
        return "generate" if state.get("hits") else "no_evidence"

    g = StateGraph(RagState)
    for name, fn in [("guard", guard), ("rewrite", rewrite), ("retrieve", retrieve),
                     ("rerank", rerank), ("generate", generate), ("no_evidence", no_evidence)]:
        g.add_node(name, timed_node(name, fn))
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, {"end": END, "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", after_rerank,
                            {"generate": "generate", "no_evidence": "no_evidence"})
    g.add_edge("generate", END)
    g.add_edge("no_evidence", END)
    return g.compile()
