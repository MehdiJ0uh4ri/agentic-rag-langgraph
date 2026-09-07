"""Evaluators.

Deterministic checks come first and are the ones to trust: whether the agent chose
to answer or refuse, and whether its citations are real. LLM judges only cover what
string comparison cannot -- semantic correctness and refusal quality.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_rag.corpus import load_corpus
from agentic_rag.llm import chat

_JUDGE_EFFORT = "medium"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- #
# Deterministic
# --------------------------------------------------------------------------- #

def refusal_decision(outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent answer when it should have, and refuse when it should have?

    This is the headline metric. A confident wrong answer to an unanswerable
    question is the failure this whole architecture exists to prevent.
    """
    actual = "refuse" if outputs.get("refusal") else "answer"
    expected = reference_outputs["expected"]
    correct = actual == expected
    return {
        "key": "refusal_decision",
        "score": int(correct),
        "comment": "expected " + expected + ", got " + actual,
    }


def citation_validity(outputs: dict, **_: Any) -> dict:
    """Every citation must name a real chunk and quote it verbatim."""
    answer = outputs.get("answer")
    if not answer:
        return {"key": "citation_validity", "score": None, "comment": "no answer (refused)"}

    corpus = {c.id: _norm(c.text) for c in load_corpus()}
    citations = answer.get("citations", [])
    if not citations:
        return {"key": "citation_validity", "score": 0, "comment": "answer had no citations"}

    bad = []
    for cit in citations:
        chunk_id, quote = cit["chunk_id"], _norm(cit["quote"])
        if chunk_id.startswith("web#"):
            continue  # web chunks are not in the local corpus; skip rather than fail
        if chunk_id not in corpus:
            bad.append(chunk_id + ": unknown chunk")
        elif quote not in corpus[chunk_id]:
            bad.append(chunk_id + ": quote not verbatim")
    score = 1 - len(bad) / len(citations)
    return {"key": "citation_validity", "score": score, "comment": "; ".join(bad) or "all citations verbatim"}


def context_recall(outputs: dict, reference_outputs: dict) -> dict:
    """Did retrieval surface the documents that actually hold the answer?"""
    expected_docs = reference_outputs.get("expected_docs") or []
    if not expected_docs:
        return {"key": "context_recall", "score": None, "comment": "no gold documents"}
    retrieved = {origin for origin in outputs.get("retrieved_origins", [])}
    hit = sum(1 for doc in expected_docs if doc in retrieved)
    return {
        "key": "context_recall",
        "score": hit / len(expected_docs),
        "comment": str(hit) + "/" + str(len(expected_docs)) + " gold docs retrieved",
    }


def decomposition_shape(outputs: dict, **_: Any) -> dict:
    """Cheap structural check on the plan: no empty, duplicate or pronoun-laden sub-questions."""
    plan = outputs.get("plan")
    if not plan:
        return {"key": "decomposition_shape", "score": 0, "comment": "no plan produced"}
    subs = plan.get("sub_questions", [])
    texts = [_norm(s["text"]) for s in subs]
    problems = []
    if not subs:
        problems.append("empty plan")
    if len(set(texts)) != len(texts):
        problems.append("duplicate sub-questions")
    dangling = [t for t in texts if re.match(r"^(it|they|this|that|those)\b", t)]
    if dangling:
        problems.append(str(len(dangling)) + " sub-question(s) start with a dangling pronoun")
    return {
        "key": "decomposition_shape",
        "score": int(not problems),
        "comment": "; ".join(problems) or str(len(subs)) + " well-formed sub-questions",
    }


# --------------------------------------------------------------------------- #
# LLM judges
# --------------------------------------------------------------------------- #

class _Judgement(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _judge(system: str, user: str) -> _Judgement:
    llm = chat(effort=_JUDGE_EFFORT, max_tokens=8000, tag="judge")
    return llm.with_structured_output(_Judgement).invoke(
        [SystemMessage(system), HumanMessage(user)]
    )


_CORRECTNESS = """You score a RAG answer against a reference answer.

1.0  every fact in the reference is present and nothing contradicts it.
0.5  partially correct, or correct but omitting a material condition.
0.0  contradicts the reference, or answers a different question.

Extra correct detail is not penalised. Different wording is not penalised. A missing \
numeric threshold or deadline IS penalised."""


def answer_correctness(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer")
    if not answer:
        return {"key": "answer_correctness", "score": None, "comment": "refused"}
    verdict = _judge(
        _CORRECTNESS,
        "Question: " + outputs["question"]
        + "\n\nReference answer:\n" + reference_outputs["reference"]
        + "\n\nAgent answer:\n" + answer["answer"],
    )
    return {"key": "answer_correctness", "score": verdict.score, "comment": verdict.reasoning}


_REFUSAL_QUALITY = """You score the quality of a RAG system's refusal.

1.0  names precisely what is missing, reports any partial findings that ARE supported, \
and does not smuggle in a guessed answer.
0.5  refuses correctly but vaguely, or omits partial findings it clearly had.
0.0  hedged non-answer, blames the user, or hints at an answer it cannot support.

A refusal that quietly states the answer anyway scores 0.0."""


def refusal_quality(outputs: dict, reference_outputs: dict) -> dict:
    refusal = outputs.get("refusal")
    if not refusal:
        return {"key": "refusal_quality", "score": None, "comment": "answered"}
    verdict = _judge(
        _REFUSAL_QUALITY,
        "Question: " + outputs["question"]
        + "\n\nWhy the corpus cannot answer it:\n" + reference_outputs["reference"]
        + "\n\nAgent refusal:\n" + str(refusal),
    )
    return {"key": "refusal_quality", "score": verdict.score, "comment": verdict.reasoning}


DETERMINISTIC = [refusal_decision, citation_validity, context_recall, decomposition_shape]
JUDGES = [answer_correctness, refusal_quality]
ALL = DETERMINISTIC + JUDGES
