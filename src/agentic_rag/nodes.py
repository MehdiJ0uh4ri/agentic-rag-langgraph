"""Graph nodes. Each one is a pure `state -> partial state` function."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag import prompts
from agentic_rag.config import settings
from agentic_rag.corpus import Chunk
from agentic_rag.llm import chat
from agentic_rag.retrieval import retrieve
from agentic_rag.schemas import (
    Answer,
    ContextGrade,
    Groundedness,
    QueryPlan,
    Refusal,
)
from agentic_rag.state import AgentState


def _render(chunks: list[Chunk]) -> str:
    return "\n\n---\n\n".join(c.render() for c in chunks) or "(no chunks retrieved)"


# --------------------------------------------------------------------------- #
# 1. Plan
# --------------------------------------------------------------------------- #

def plan_node(state: AgentState) -> AgentState:
    llm = chat(effort=settings.planner_effort, tag="planner")
    plan: QueryPlan = llm.with_structured_output(QueryPlan).invoke([
        SystemMessage(prompts.PLANNER),
        HumanMessage("Question: " + state["question"]),
    ])
    plan.sub_questions = plan.sub_questions[: settings.max_sub_questions]
    return {
        "plan": plan,
        "round": 0,
        "notes": ["planned " + str(len(plan.sub_questions)) + " sub-question(s)"],
    }


# --------------------------------------------------------------------------- #
# 2. Retrieve
# --------------------------------------------------------------------------- #

def retrieve_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    followups = state.get("followups") or {}
    round_no = state.get("round", 0)

    fresh: dict[str, list[Chunk]] = {}
    for sq in plan.sub_questions:
        if round_no > 0 and sq.id not in followups:
            continue  # this sub-question is already covered; don't re-search it
        query = followups.get(sq.id, sq.text)
        sources = list(sq.sources) or ["vector", "lexical"]
        if not settings.enable_web:
            sources = [s for s in sources if s != "web"]
        fresh[sq.id] = retrieve(query, sources)

    retrieved_now = sum(len(v) for v in fresh.values())
    return {
        "retrieved": fresh,
        "notes": ["round " + str(round_no) + ": retrieved " + str(retrieved_now) + " chunk(s)"],
    }


# --------------------------------------------------------------------------- #
# 3. Grade the context
# --------------------------------------------------------------------------- #

def grade_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    retrieved = state.get("retrieved") or {}
    grader = chat(effort=settings.grader_effort, max_tokens=8000, tag="grader")
    grader = grader.with_structured_output(ContextGrade)

    batch = []
    order = []
    for sq in plan.sub_questions:
        chunks = retrieved.get(sq.id, [])
        order.append(sq.id)
        batch.append([
            SystemMessage(prompts.CONTEXT_GRADER),
            HumanMessage(
                "Sub-question id: " + sq.id
                + "\nSub-question: " + sq.text
                + "\n\nRetrieved chunks:\n" + _render(chunks)
            ),
        ])

    # `.batch` fans the grader calls out concurrently and keeps LangSmith parenting.
    results = grader.batch(batch, config={"max_concurrency": 4}, return_exceptions=True)

    grades: list[ContextGrade] = []
    for sq_id, result in zip(order, results, strict=True):
        if isinstance(result, BaseException):
            # A grader that fails is treated as "insufficient" -- never as a pass.
            grades.append(ContextGrade(sub_question_id=sq_id, grades=[], sufficiency=0.0,
                                       missing="grader failed: " + str(result)))
            continue
        result.sub_question_id = sq_id
        grades.append(result)

    covered = [g for g in grades if _is_covered(g)]
    coverage = len(covered) / max(1, len(grades))
    followups = {
        g.sub_question_id: g.followup_query
        for g in grades
        if not _is_covered(g) and g.followup_query
    }
    return {
        "grades": grades,
        "coverage": coverage,
        "followups": followups,
        "notes": ["coverage " + format(coverage, ".2f")],
    }


def _is_covered(grade: ContextGrade) -> bool:
    relevant = sum(1 for g in grade.grades if g.relevant)
    return relevant >= settings.min_relevant_chunks and grade.sufficiency >= settings.min_sufficiency


# --------------------------------------------------------------------------- #
# 4. Decide
# --------------------------------------------------------------------------- #

def decide(state: AgentState) -> str:
    """Conditional edge: answer, retry retrieval, or refuse."""
    plan = state.get("plan")
    if plan is not None and not plan.is_answerable_in_principle:
        return "refuse"
    if state.get("coverage", 0.0) >= settings.min_coverage:
        return "answer"
    if state.get("round", 0) + 1 < settings.max_retrieval_rounds and state.get("followups"):
        return "retry"
    return "refuse"


def bump_round(state: AgentState) -> AgentState:
    return {"round": state.get("round", 0) + 1}


# --------------------------------------------------------------------------- #
# 5. Answer / refuse / verify
# --------------------------------------------------------------------------- #

def _all_chunks(state: AgentState) -> list[Chunk]:
    seen: dict[str, Chunk] = {}
    for chunks in (state.get("retrieved") or {}).values():
        for c in chunks:
            seen.setdefault(c.id, c)
    return list(seen.values())


def answer_node(state: AgentState) -> AgentState:
    llm = chat(effort=settings.answer_effort, tag="answerer")
    answer: Answer = llm.with_structured_output(Answer).invoke([
        SystemMessage(prompts.ANSWERER),
        HumanMessage(
            "Question: " + state["question"]
            + "\n\nContext:\n" + _render(_all_chunks(state))
        ),
    ])
    return {"answer": answer, "notes": ["drafted answer with " + str(len(answer.citations)) + " citation(s)"]}


def verify_node(state: AgentState) -> AgentState:
    answer = state["answer"]
    by_id = {c.id: c for c in _all_chunks(state)}
    cited = "\n\n".join(
        "[" + cit.chunk_id + "] quote: " + cit.quote
        + "\nsource text: " + by_id.get(cit.chunk_id, Chunk(cit.chunk_id, "(unknown chunk)", "", "", "")).text
        for cit in answer.citations
    )
    llm = chat(effort=settings.grader_effort, max_tokens=8000, tag="groundedness")
    verdict: Groundedness = llm.with_structured_output(Groundedness).invoke([
        SystemMessage(prompts.GROUNDEDNESS),
        HumanMessage("Answer:\n" + answer.answer + "\n\nCitations:\n" + cited),
    ])
    return {
        "groundedness": verdict,
        "notes": ["grounded=" + str(verdict.grounded)],
    }


def after_verify(state: AgentState) -> str:
    verdict = state.get("groundedness")
    return "ok" if verdict is None or verdict.grounded else "refuse"


def refuse_node(state: AgentState) -> AgentState:
    plan = state.get("plan")
    grades = state.get("grades") or []
    gaps = "\n".join(
        "- " + g.sub_question_id + ": sufficiency=" + format(g.sufficiency, ".2f")
        + " missing=" + (g.missing or "n/a")
        for g in grades
    ) or "- no retrieval was attempted"

    verdict = state.get("groundedness")
    draft = state.get("answer")
    extra = ""
    if verdict is not None and not verdict.grounded and draft is not None:
        extra = (
            "\n\nA draft answer was rejected as ungrounded. Unsupported claims:\n"
            + "\n".join("- " + c for c in verdict.unsupported_claims)
        )

    llm = chat(effort=settings.grader_effort, tag="refuser")
    refusal: Refusal = llm.with_structured_output(Refusal).invoke([
        SystemMessage(prompts.REFUSER),
        HumanMessage(
            "Question: " + state["question"]
            + "\n\nIntent: " + (plan.intent if plan else "unknown")
            + "\n\nPer-sub-question gaps:\n" + gaps
            + extra
            + "\n\nRetrieved context (all of it):\n" + _render(_all_chunks(state))
        ),
    ])
    return {"refusal": refusal, "decision": "refuse", "notes": ["refused"]}


def finalize_answer(state: AgentState) -> AgentState:
    return {"decision": "answer"}
