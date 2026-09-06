"""Pydantic schemas used as structured outputs for planning, grading and answering.

Every schema here is bound with `.with_structured_output(..., strict=True)`, so the
model returns validated JSON rather than prose we have to parse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One atomic, independently answerable piece of the user's question."""

    id: str = Field(description="Stable short id, e.g. 'q1'.")
    text: str = Field(description="A self-contained question. No pronouns referring to the parent query.")
    rationale: str = Field(description="Why answering this is needed to answer the parent query.")
    sources: list[Literal["vector", "lexical", "web"]] = Field(
        default_factory=lambda: ["vector", "lexical"],
        description=(
            "Which sources are worth searching. Use 'web' only for facts that are "
            "recent, external, or clearly outside an internal knowledge base."
        ),
    )


class QueryPlan(BaseModel):
    """Decomposition of the user's query."""

    intent: str = Field(description="One sentence: what the user actually wants.")
    is_answerable_in_principle: bool = Field(
        description="False for questions no document corpus could answer (opinion, prediction, personal data)."
    )
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=6)
    search_queries: list[str] = Field(
        default_factory=list,
        description="Extra keyword-style queries for lexical search, beyond the sub-question text.",
    )


class ChunkGrade(BaseModel):
    """Relevance verdict for a single retrieved chunk."""

    chunk_id: str
    relevant: bool = Field(description="True only if the chunk contains evidence bearing on the sub-question.")
    reason: str = Field(description="One short clause. Cite the deciding phrase from the chunk when relevant.")


class ContextGrade(BaseModel):
    """Whether the retrieved context can answer one sub-question."""

    sub_question_id: str
    grades: list[ChunkGrade]
    sufficiency: float = Field(
        ge=0.0, le=1.0,
        description=(
            "0.0 = nothing useful. 0.5 = partial, would force guessing. "
            "1.0 = the sub-question can be answered fully and cited."
        ),
    )
    missing: str = Field(
        default="",
        description="What evidence is absent. Empty when sufficiency is high.",
    )
    followup_query: str = Field(
        default="",
        description="A rewritten search query likely to surface the missing evidence. Empty if none would help.",
    )


class Citation(BaseModel):
    chunk_id: str
    quote: str = Field(description="Verbatim span from the chunk that supports the claim.")


class Answer(BaseModel):
    """The grounded answer."""

    answer: str = Field(description="Markdown. Every factual claim must be traceable to a citation.")
    citations: list[Citation] = Field(min_length=1)
    caveats: str = Field(default="", description="What the answer does not cover, if anything.")


class Refusal(BaseModel):
    """A principled refusal when context is insufficient."""

    reason: str = Field(description="Plain-language explanation of what the corpus does not cover.")
    missing_evidence: list[str] = Field(description="Concrete facts that would be needed.")
    partial_findings: str = Field(
        default="",
        description="Anything genuinely established by the retrieved context. Empty if nothing was.",
    )
    suggested_rephrasing: str = Field(default="")


class Groundedness(BaseModel):
    """Post-hoc check that the drafted answer is supported by its citations."""

    grounded: bool = Field(description="False if any claim goes beyond the cited spans.")
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
