"""Guardrails: cheap, deterministic, run on every request.

Input : length cap, prompt-injection heuristics (OWASP LLM01), PII redaction for logs.
Output: groundedness check - every citation must point to a chunk that was really
        retrieved (blocks fabricated sources), and a non-refusal answer must cite something.

Heuristics are a first layer, not a silver bullet: pair with model-side defences (the
prompt marks context as untrusted) and, for high-risk apps, a classifier (Llama Guard,
Prompt Guard) - see docs/security.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from open_rag.models import Answer

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(system|previous) (prompt|instructions)",
    r"you are now (?!able)",
    r"reveal (your |the )?(system )?prompt",
    r"ignore (toutes )?(les )?instructions (précédentes|ci-dessus)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.IGNORECASE)
_PII = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"\+?\d[\d .-]{8,}\d"),
}


class GuardrailError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def check_input(question: str, max_chars: int) -> str:
    q = question.strip()
    if not q:
        raise GuardrailError("empty", "Question is empty.")
    if len(q) > max_chars:
        raise GuardrailError("too_long", f"Question exceeds {max_chars} characters.")
    if _INJECTION_RE.search(q):
        raise GuardrailError("prompt_injection", "Potential prompt injection detected.")
    return q


def check_history(history: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    """History is client-supplied, so it is as untrusted as the question: keep only the last
    turns and apply the same injection screen (a forged 'assistant' turn is a known vector)."""
    kept = history[-max_messages:] if max_messages > 0 else []
    for m in kept:
        if _INJECTION_RE.search(m["content"]):
            raise GuardrailError("prompt_injection", "Potential prompt injection in history.")
    return kept


def redact_pii(text: str) -> str:
    for label, rx in _PII.items():
        text = rx.sub(f"<{label}>", text)
    return text


@dataclass
class GroundingReport:
    grounded: bool
    reason: str
    valid_ids: list[str]


def check_grounding(answer: Answer, retrieved_ids: set[str]) -> GroundingReport:
    if answer.insufficient_context:
        return GroundingReport(True, "explicit_refusal", [])
    valid = [i for i in answer.cited_ids if i in retrieved_ids]
    if not answer.cited_ids:
        return GroundingReport(False, "no_citation", [])
    if len(valid) != len(answer.cited_ids):
        return GroundingReport(False, "fabricated_citation", valid)
    return GroundingReport(True, "ok", valid)
