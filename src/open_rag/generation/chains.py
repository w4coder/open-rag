"""LLM chains (LCEL) with offline extractive fallbacks.

Each function returns a Runnable so that, when LangSmith tracing is on, every step appears
as a named span (`run_name`), which is what makes production debugging possible.
Real models come from `providers.structured_llm` / `text_llm`: timeout + retries + fallback.
"""
from __future__ import annotations

import math
import re

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from open_rag.config import Settings
from open_rag.generation.prompts import (
    ANSWER_PROMPT,
    CONTEXTUAL_PROMPT,
    RERANK_PROMPT,
    REWRITE_PROMPT,
)
from open_rag.models import Answer
from open_rag.providers import structured_llm, text_llm, tokenize

STOPWORDS = frozenset(
    ["le", "la", "les", "un", "une", "des", "du", "de", "et", "en", "est", "the", "a", "an", "of", "to", "in", "is", "are", "what", "how", "do", "does", "pour", "que", "qui", "quoi", "comment", "quel", "quelle", "dans", "sur", "au", "aux", "ce", "cette", "ces", "il", "elle", "on", "and", "or", "for"]
)
REFUSAL = "I don't know based on the provided documents."


def format_context(hits) -> str:
    # Passages are numbered [1]..[n], not labelled by chunk UUID: models copy "[3]" reliably,
    # 36-char ids less so, and a mistyped id is (rightly) rejected by the grounding guardrail.
    return "\n\n".join(f"[{i}] (source: {h.source} | {h.section})\n{h.parent_text}"
                       for i, h in enumerate(hits, 1))


def resolve_citations(ans: Answer, hits) -> Answer:
    """Map cited passage numbers back to chunk ids. Anything that is not a valid number is
    kept as-is, so the grounding guardrail still flags it as fabricated."""
    ids = []
    for c in ans.cited_ids:
        n = c.strip().strip("[]")
        ids.append(hits[int(n) - 1].chunk_id if n.isdigit() and 1 <= int(n) <= len(hits) else c)
    return ans.model_copy(update={"cited_ids": list(dict.fromkeys(ids))})


def _content(m) -> str:
    return m.content.strip()


# ---- query rewriting ---------------------------------------------------------------------
def build_rewriter(settings: Settings) -> Runnable:
    llm = text_llm(settings)
    if llm is None:
        return RunnableLambda(lambda x: x["question"]).with_config(run_name="rewrite_fake")
    return (REWRITE_PROMPT | llm | RunnableLambda(_content)).with_config(run_name="rewrite_query")


# ---- answer generation -------------------------------------------------------------------
def extractive_answer(inp: dict) -> Answer:
    """Offline stand-in for the LLM: best-matching sentence (IDF-weighted lexical overlap)
    plus the sentence that follows it in the same chunk ("sentence window": answers often
    continue in the next sentence, e.g. 'The default limit is X. The Business plan raises it to Y.')
    """
    q = {t for t in tokenize(inp["question"]) if t not in STOPWORDS}
    sents: list[tuple[str, str, set[str]]] = []
    for h in inp["hits"]:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", h.text):
            toks = set(tokenize(sent))
            if len(toks) >= 4:
                sents.append((sent.strip(), h.chunk_id, toks))
    # IDF weighting: a rare term ("business") must outweigh a common one ("rate").
    n = len(sents) or 1
    weight = {t: math.log(1 + n / max(1, sum(t in s[2] for s in sents))) for t in q}
    total = sum(weight.values()) or 1.0
    scores = [sum(weight[t] for t in q & toks) / total for _, _, toks in sents]
    if not scores or max(scores) < 0.34:
        return Answer(answer=REFUSAL, insufficient_context=True)
    i = max(range(len(scores)), key=scores.__getitem__)
    picked = [sents[i]]
    if i + 1 < len(sents) and sents[i + 1][1] == sents[i][1]:
        picked.append(sents[i + 1])
    return Answer(answer=" ".join(s[0] for s in picked), cited_ids=[sents[i][1]])


def build_answerer(settings: Settings) -> Runnable:
    """Runnable[{question, hits}] -> Answer"""
    llm = structured_llm(settings, Answer)
    if llm is None:
        return RunnableLambda(extractive_answer).with_config(run_name="answer_fake")
    generate = (
        RunnableLambda(lambda x: {"question": x["question"], "context": format_context(x["hits"])})
        | ANSWER_PROMPT
        | llm
    )
    return (
        RunnablePassthrough.assign(answer=generate)
        | RunnableLambda(lambda x: resolve_citations(x["answer"], x["hits"]))
    ).with_config(run_name="generate_answer")


# ---- contextual retrieval (enrichment) ---------------------------------------------------
def build_contextualizer(settings: Settings) -> Runnable | None:
    llm = text_llm(settings)
    if llm is None or not settings.contextual_enrichment:
        return None
    return (CONTEXTUAL_PROMPT | llm | RunnableLambda(_content)).with_config(run_name="contextualize")


# ---- LLM reranker ------------------------------------------------------------------------
class _PassageScore(BaseModel):
    id: int = Field(description="Passage number, as in [n].")
    score: float = Field(description="0 (useless) to 10 (directly answers).")


class _Scores(BaseModel):
    # A list, not dict[str, float]: OpenAI strict structured output rejects free-form keys.
    scores: list[_PassageScore]


def build_llm_scorer(settings: Settings) -> Runnable | None:
    llm = structured_llm(settings, _Scores)
    return None if llm is None else (RERANK_PROMPT | llm).with_config(run_name="llm_rerank")
