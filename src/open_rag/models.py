"""Domain / API models shared across layers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    source: str
    section: str = ""
    snippet: str


class Answer(BaseModel):
    """Structured generation output (enforced via with_structured_output).

    WHY: structured output makes citations machine-checkable (guardrail +
    evaluators can verify every cited id was really retrieved).
    """

    answer: str = Field(description="Answer grounded ONLY in the provided context.")
    cited_ids: list[str] = Field(default_factory=list, description="Ids of chunks used.")
    insufficient_context: bool = Field(
        default=False, description="True if the context cannot answer the question."
    )


class Message(BaseModel):
    """One prior conversation turn. Roles are constrained: a client cannot inject `system`."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(max_length=8000)  # hard cap; the guardrail applies the tuned limit
    history: list[Message] = Field(default_factory=list, max_length=50)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    cached: bool = False
    trace: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: str
    status: str
