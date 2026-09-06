"""Claude client factories.

Two surfaces are used deliberately:

* `chat(...)` -> LangChain `ChatAnthropic`, so LangGraph nodes get structured
  output binding and automatic LangSmith spans.
* `raw_client()` -> the official `anthropic` SDK, used for the server-side
  `web_search` tool, which has no LangChain equivalent.
"""

from __future__ import annotations

from functools import lru_cache

import anthropic
from langchain_anthropic import ChatAnthropic

from agentic_rag.config import settings

# Thinking stays adaptive everywhere. Disabling it on Opus 5 lets tool calls leak
# into visible text, which silently breaks structured output; lowering `effort`
# is the correct cost lever instead.
_THINKING = {"type": "adaptive"}


@lru_cache(maxsize=8)
def chat(effort: str = "high", max_tokens: int = 16_000, tag: str = "llm") -> ChatAnthropic:
    """A configured Claude chat model. Cached per (effort, max_tokens, tag)."""
    return ChatAnthropic(
        model=settings.model,
        max_tokens=max_tokens,
        thinking=_THINKING,
        model_kwargs={"output_config": {"effort": effort}},
        tags=[tag],
    )


@lru_cache(maxsize=1)
def raw_client() -> anthropic.Anthropic:
    """Official SDK client. Credentials resolve from env or an `ant auth login` profile."""
    return anthropic.Anthropic()
