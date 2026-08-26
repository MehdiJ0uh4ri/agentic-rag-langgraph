"""Runtime configuration, resolved once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # One model for the whole agent. Graders run the same model at low effort --
    # cheaper without swapping in a weaker judge.
    model: str = os.getenv("ARAG_MODEL", "claude-opus-5")
    planner_effort: str = os.getenv("ARAG_PLANNER_EFFORT", "high")
    grader_effort: str = os.getenv("ARAG_GRADER_EFFORT", "low")
    answer_effort: str = os.getenv("ARAG_ANSWER_EFFORT", "high")

    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    index_dir: Path = REPO_ROOT / ".index"

    max_sub_questions: int = int(os.getenv("ARAG_MAX_SUB_QUESTIONS", "4"))
    max_retrieval_rounds: int = int(os.getenv("ARAG_MAX_RETRIEVAL_ROUNDS", "2"))
    top_k_per_source: int = int(os.getenv("ARAG_TOP_K", "4"))

    enable_web: bool = _bool("ARAG_ENABLE_WEB", True)
    web_allowed_domains: tuple[str, ...] = field(default_factory=tuple)

    # A sub-question is answerable when at least this many of its chunks are
    # graded relevant, and the grader's own confidence clears the floor.
    min_relevant_chunks: int = int(os.getenv("ARAG_MIN_RELEVANT", "1"))
    min_sufficiency: float = float(os.getenv("ARAG_MIN_SUFFICIENCY", "0.6"))
    # Refuse unless this share of sub-questions is covered.
    min_coverage: float = float(os.getenv("ARAG_MIN_COVERAGE", "0.75"))

    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "agentic-rag")

    @property
    def tracing_enabled(self) -> bool:
        return _bool("LANGSMITH_TRACING", False) and bool(os.getenv("LANGSMITH_API_KEY"))


settings = Settings()
