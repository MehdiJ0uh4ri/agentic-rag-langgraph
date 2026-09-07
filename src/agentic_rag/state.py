"""Graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from agentic_rag.corpus import Chunk
from agentic_rag.schemas import Answer, ContextGrade, Groundedness, QueryPlan, Refusal


def merge_chunks(
    left: dict[str, list[Chunk]] | None, right: dict[str, list[Chunk]] | None
) -> dict[str, list[Chunk]]:
    """Union retrieved chunks per sub-question, de-duplicated by chunk id.

    A reducer rather than an overwrite so a second retrieval round adds to what the
    first found instead of discarding it.
    """
    out: dict[str, list[Chunk]] = {k: list(v) for k, v in (left or {}).items()}
    for sq_id, chunks in (right or {}).items():
        seen = {c.id for c in out.get(sq_id, [])}
        out.setdefault(sq_id, []).extend(c for c in chunks if c.id not in seen)
    return out


class AgentState(TypedDict, total=False):
    question: str

    plan: QueryPlan
    retrieved: Annotated[dict[str, list[Chunk]], merge_chunks]
    grades: list[ContextGrade]

    # Sub-question id -> rewritten query, produced by the grader for round N+1.
    followups: dict[str, str]
    round: int
    coverage: float

    answer: Answer
    refusal: Refusal
    groundedness: Groundedness

    decision: Literal["answer", "retry", "refuse"]
    notes: Annotated[list[str], operator.add]
