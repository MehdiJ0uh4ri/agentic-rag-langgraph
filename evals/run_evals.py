"""Run the eval suite against LangSmith.

    python evals/run_evals.py                  # full suite
    python evals/run_evals.py --no-judges      # deterministic metrics only (cheap)
    python evals/run_evals.py --experiment v2-stricter-grader
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langsmith import (
    Client,
    evaluate,
)

from agentic_rag.config import settings
from agentic_rag.graph import run
from evals import evaluators
from evals.dataset import to_langsmith


def target(inputs: dict) -> dict:
    """Adapter: LangSmith example -> flat dict the evaluators can read."""
    state = run(inputs["question"])
    answer = state.get("answer")
    refusal = state.get("refusal")
    plan = state.get("plan")
    origins = {
        chunk.origin
        for chunks in (state.get("retrieved") or {}).values()
        for chunk in chunks
    }
    return {
        "question": inputs["question"],
        "answer": answer.model_dump() if answer else None,
        "refusal": refusal.model_dump() if refusal else None,
        "plan": plan.model_dump() if plan else None,
        "coverage": state.get("coverage"),
        "rounds": state.get("round", 0) + 1,
        "grounded": state["groundedness"].grounded if state.get("groundedness") else None,
        "retrieved_origins": sorted(origins),
        "notes": state.get("notes", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--no-judges", action="store_true", help="skip the LLM-judge evaluators")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    if not settings.tracing_enabled:
        print("LANGSMITH_TRACING / LANGSMITH_API_KEY are not set -- results will not be recorded.")
        return 1

    client = Client()
    dataset_name = to_langsmith(client)
    metrics = evaluators.DETERMINISTIC if args.no_judges else evaluators.ALL

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=metrics,
        experiment_prefix=args.experiment,
        max_concurrency=args.concurrency,
        metadata={
            "model": settings.model,
            "grader_effort": settings.grader_effort,
            "min_coverage": settings.min_coverage,
            "min_sufficiency": settings.min_sufficiency,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
        },
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
