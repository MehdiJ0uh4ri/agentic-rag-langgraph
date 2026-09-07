"""The LangGraph wiring.

    plan ─┬─(unanswerable)─────────────────────────────► refuse ► END
          └► retrieve ► grade ─┬─(coverage ok)──► answer ► verify ─┬─(grounded)─► END
                               │                                   └─(hallucinated)─┐
                               ├─(gaps + rounds left)─► bump ─┐                     │
                               │                              └──► retrieve         │
                               └─(gaps, no rounds left)──────────────────► refuse ◄─┘
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from agentic_rag import nodes
from agentic_rag.config import settings
from agentic_rag.state import AgentState


def _route_after_plan(state: AgentState) -> str:
    plan = state.get("plan")
    return "refuse" if plan is not None and not plan.is_answerable_in_principle else "retrieve"


def build_graph() -> Any:
    g = StateGraph(AgentState)

    g.add_node("plan", nodes.plan_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("grade", nodes.grade_node)
    g.add_node("bump", nodes.bump_round)
    g.add_node("answer", nodes.answer_node)
    g.add_node("verify", nodes.verify_node)
    g.add_node("finalize", nodes.finalize_answer)
    g.add_node("refuse", nodes.refuse_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _route_after_plan,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.decide,
                            {"answer": "answer", "retry": "bump", "refuse": "refuse"})
    g.add_edge("bump", "retrieve")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", nodes.after_verify,
                            {"ok": "finalize", "refuse": "refuse"})
    g.add_edge("finalize", END)
    g.add_edge("refuse", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph() -> Any:
    return build_graph()


def run(question: str, **config: Any) -> AgentState:
    """Run the agent once. Every LLM and retriever call lands in one LangSmith trace."""
    run_config = {
        "run_name": "agentic-rag",
        "metadata": {
            "model": settings.model,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
            "min_coverage": settings.min_coverage,
        },
        "tags": ["agentic-rag"],
        **config,
    }
    return get_graph().invoke({"question": question, "notes": []}, config=run_config)
