"""Offline tests. No API key required -- every LLM node is stubbed.

These pin the control flow (when the agent answers, retries, or refuses), which is
the part of the system that must not regress silently.
"""

from __future__ import annotations

import pytest

from agentic_rag import graph as graph_mod
from agentic_rag import nodes
from agentic_rag.config import settings
from agentic_rag.corpus import Chunk, load_corpus
from agentic_rag.schemas import (
    Answer,
    ChunkGrade,
    Citation,
    ContextGrade,
    Groundedness,
    QueryPlan,
    Refusal,
    SubQuestion,
)
from agentic_rag.state import merge_chunks


def chunk(cid: str, text: str = "body", origin: str = "handbook/expenses.md") -> Chunk:
    return Chunk(id=cid, text=text, source="vector", origin=origin, title="t", score=1.0)


# --------------------------------------------------------------------------- #
# Corpus + retrieval
# --------------------------------------------------------------------------- #

def test_corpus_loads_and_chunks():
    chunks = load_corpus()
    assert chunks, "sample corpus should not be empty"
    assert all(c.id.count("#") == 1 for c in chunks)
    assert any("expenses.md" in c.origin for c in chunks)


def test_lexical_retrieval_finds_exact_terms():
    from agentic_rag.retrieval import lexical_source

    hits = lexical_source("break-glass account rotated every 90 days", k=3)
    assert hits
    assert any("access-control" in h.origin for h in hits)


def test_fan_out_dedupes_across_sources():
    from agentic_rag.retrieval import retrieve

    fused = retrieve("annual leave carry over", ["vector", "lexical"], k=4)
    assert len(fused) == len({c.id for c in fused}), "fusion must de-duplicate by chunk id"


# --------------------------------------------------------------------------- #
# State reducer
# --------------------------------------------------------------------------- #

def test_merge_chunks_unions_rounds():
    first = {"q1": [chunk("a"), chunk("b")]}
    second = {"q1": [chunk("b"), chunk("c")], "q2": [chunk("d")]}
    merged = merge_chunks(first, second)
    assert [c.id for c in merged["q1"]] == ["a", "b", "c"]
    assert [c.id for c in merged["q2"]] == ["d"]


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def covered(sq_id: str = "q1") -> ContextGrade:
    return ContextGrade(
        sub_question_id=sq_id,
        grades=[ChunkGrade(chunk_id="a", relevant=True, reason="states the threshold")],
        sufficiency=0.9,
    )


def uncovered(sq_id: str = "q1", followup: str = "rephrased query") -> ContextGrade:
    return ContextGrade(
        sub_question_id=sq_id,
        grades=[ChunkGrade(chunk_id="a", relevant=False, reason="off topic")],
        sufficiency=0.1,
        missing="the actual rate",
        followup_query=followup,
    )


def plan(answerable: bool = True) -> QueryPlan:
    return QueryPlan(
        intent="test",
        is_answerable_in_principle=answerable,
        sub_questions=[SubQuestion(id="q1", text="What is X?", rationale="needed")],
    )


def test_decide_answers_on_full_coverage():
    assert nodes.decide({"plan": plan(), "coverage": 1.0, "round": 0}) == "answer"


def test_decide_retries_when_followup_available():
    state = {"plan": plan(), "coverage": 0.0, "round": 0, "followups": {"q1": "retry me"}}
    assert nodes.decide(state) == "retry"


def test_decide_refuses_when_rounds_exhausted():
    state = {
        "plan": plan(),
        "coverage": 0.0,
        "round": settings.max_retrieval_rounds - 1,
        "followups": {"q1": "retry me"},
    }
    assert nodes.decide(state) == "refuse"


def test_decide_refuses_unaskable_without_retrieving():
    assert nodes.decide({"plan": plan(answerable=False), "coverage": 1.0}) == "refuse"


def test_grader_failure_counts_as_insufficient():
    assert nodes._is_covered(uncovered()) is False
    assert nodes._is_covered(covered()) is True


def test_ungrounded_answer_routes_to_refusal():
    verdict = Groundedness(grounded=False, unsupported_claims=["invented a deadline"], confidence=0.9)
    assert nodes.after_verify({"groundedness": verdict}) == "refuse"
    assert nodes.after_verify({"groundedness": Groundedness(grounded=True, confidence=0.9)}) == "ok"


# --------------------------------------------------------------------------- #
# End-to-end with stubbed LLM nodes
# --------------------------------------------------------------------------- #

@pytest.fixture
def stub(monkeypatch):
    """Replace every LLM-backed node with a deterministic stand-in."""
    calls: dict[str, int] = {}

    def counted(name, fn):
        def wrapper(state):
            calls[name] = calls.get(name, 0) + 1
            return fn(state)
        return wrapper

    monkeypatch.setattr(nodes, "plan_node", counted("plan", lambda s: {"plan": plan(), "round": 0, "notes": []}))
    monkeypatch.setattr(nodes, "retrieve_node", counted("retrieve", lambda s: {"retrieved": {"q1": [chunk("a")]}, "notes": []}))
    monkeypatch.setattr(
        nodes, "answer_node",
        counted("answer", lambda s: {
            "answer": Answer(answer="42.", citations=[Citation(chunk_id="a", quote="body")]),
            "notes": [],
        }),
    )
    monkeypatch.setattr(
        nodes, "refuse_node",
        counted("refuse", lambda s: {
            "refusal": Refusal(reason="not covered", missing_evidence=["the rate"]),
            "decision": "refuse",
            "notes": [],
        }),
    )
    graph_mod.get_graph.cache_clear()
    return calls


def test_end_to_end_answers_when_context_is_sufficient(stub, monkeypatch):
    monkeypatch.setattr(nodes, "grade_node", lambda s: {"grades": [covered()], "coverage": 1.0, "followups": {}, "notes": []})
    monkeypatch.setattr(nodes, "verify_node", lambda s: {"groundedness": Groundedness(grounded=True, confidence=0.95), "notes": []})
    graph_mod.get_graph.cache_clear()

    state = graph_mod.build_graph().invoke({"question": "What is X?", "notes": []})
    assert state["decision"] == "answer"
    assert state["answer"].answer == "42."
    assert stub.get("refuse", 0) == 0


def test_end_to_end_refuses_and_retries_once(stub, monkeypatch):
    monkeypatch.setattr(
        nodes, "grade_node",
        lambda s: {"grades": [uncovered()], "coverage": 0.0, "followups": {"q1": "again"}, "notes": []},
    )
    graph_mod.get_graph.cache_clear()

    state = graph_mod.build_graph().invoke({"question": "What is X?", "notes": []})
    assert state["decision"] == "refuse"
    assert state["refusal"].missing_evidence
    assert stub["retrieve"] == settings.max_retrieval_rounds, "should exhaust its retrieval budget first"
    assert stub.get("answer", 0) == 0, "must never draft an answer it cannot ground"


def test_end_to_end_refuses_when_answer_is_hallucinated(stub, monkeypatch):
    monkeypatch.setattr(nodes, "grade_node", lambda s: {"grades": [covered()], "coverage": 1.0, "followups": {}, "notes": []})
    monkeypatch.setattr(
        nodes, "verify_node",
        lambda s: {"groundedness": Groundedness(grounded=False, unsupported_claims=["made up"], confidence=0.9), "notes": []},
    )
    graph_mod.get_graph.cache_clear()

    state = graph_mod.build_graph().invoke({"question": "What is X?", "notes": []})
    assert state["decision"] == "refuse"
    assert stub["answer"] == 1


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #

def test_citation_validity_rejects_fabricated_quotes():
    from evals.evaluators import citation_validity

    real = load_corpus()[0]
    good = {"answer": {"answer": "x", "citations": [{"chunk_id": real.id, "quote": real.text[:40]}]}}
    bad = {"answer": {"answer": "x", "citations": [{"chunk_id": real.id, "quote": "this text is invented"}]}}
    assert citation_validity(good)["score"] == 1.0
    assert citation_validity(bad)["score"] == 0.0


def test_refusal_decision_scores_both_directions():
    from evals.evaluators import refusal_decision

    refused = {"refusal": {"reason": "no"}, "answer": None}
    answered = {"refusal": None, "answer": {"answer": "yes"}}
    assert refusal_decision(refused, {"expected": "refuse"})["score"] == 1
    assert refusal_decision(answered, {"expected": "refuse"})["score"] == 0
    assert refusal_decision(answered, {"expected": "answer"})["score"] == 1
