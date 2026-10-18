"""Retrieval + answer metrics.

Gold labels are *text snippets* (not chunk ids): a retrieved chunk is relevant if it contains
the gold snippet. WHY: chunk ids change every time you change the chunking strategy, which
would invalidate the dataset exactly when you most need to compare strategies.
"""
from __future__ import annotations


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def relevant_ranks(retrieved_texts: list[str], gold: str) -> list[int]:
    g = _norm(gold)
    return [i for i, t in enumerate(retrieved_texts, start=1) if g in _norm(t)]


def hit_at_k(retrieved_texts: list[str], gold: str, k: int) -> float:
    return float(any(r <= k for r in relevant_ranks(retrieved_texts, gold)))


def reciprocal_rank(retrieved_texts: list[str], gold: str) -> float:
    ranks = relevant_ranks(retrieved_texts, gold)
    return 1.0 / ranks[0] if ranks else 0.0


def answer_correct(answer: str, gold: str) -> float:
    """Lexical check: fine for factoid corpora; use an LLM judge for free-form answers."""
    if not gold:
        return 0.0
    gold_tokens = [t for t in _norm(gold).split() if len(t) > 2]
    ans = _norm(answer)
    return float(all(t in ans for t in gold_tokens))
