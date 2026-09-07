"""Multi-source retrieval: dense vectors, BM25 lexical, and Claude's web_search.

Each source is a `@traceable` function, so LangSmith shows one span per source per
sub-question alongside the LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

import numpy as np
from langsmith import traceable
from rank_bm25 import BM25Okapi

from agentic_rag.config import settings
from agentic_rag.corpus import Chunk, load_corpus
from agentic_rag.llm import raw_client

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    """Deterministic TF-IDF fallback used when `fastembed` is unavailable.

    Keeps the pipeline (and the test suite) runnable offline. Quality is lower
    than a real embedding model, but the interface and the graph are identical.
    """

    dim = 4096

    def __init__(self, corpus: list[str]) -> None:
        df = np.zeros(self.dim)
        for doc in corpus:
            for bucket in {self._bucket(t) for t in _tokenize(doc)}:
                df[bucket] += 1
        self._idf = np.log((1 + len(corpus)) / (1 + df)) + 1.0

    def _bucket(self, token: str) -> int:
        return int.from_bytes(token.encode()[:8].ljust(8, b"\0"), "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokenize(text):
                out[i, self._bucket(tok)] += 1.0
        out *= self._idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class _FastEmbedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

class Index:
    """Holds the chunked corpus plus both a dense and a lexical view of it."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [f"{c.title}\n{c.text}" for c in chunks]

        try:
            self.embedder: object = _FastEmbedder()
            self.backend = "fastembed:bge-small-en-v1.5"
        except Exception:  # noqa: BLE001 -- any import or download failure falls back
            self.embedder = _HashingEmbedder(texts)
            self.backend = "tfidf-hashing (fallback)"

        self.vectors = self.embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None

    def vector_search(self, query: str, k: int) -> list[Chunk]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        return self._top(self.vectors @ q, k, "vector")

    def lexical_search(self, query: str, k: int) -> list[Chunk]:
        if self.bm25 is None:
            return []
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
        peak = float(scores.max()) if scores.size else 0.0
        if peak > 0:
            scores = scores / peak
        return self._top(scores, k, "lexical")

    def _top(self, scores: np.ndarray, k: int, source: str) -> list[Chunk]:
        order = np.argsort(-scores)[:k]
        return [
            replace(self.chunks[i], source=source, score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index(load_corpus())


# --------------------------------------------------------------------------- #
# Traced source functions
# --------------------------------------------------------------------------- #

@traceable(name="retrieve:vector", run_type="retriever")
def vector_source(query: str, k: int | None = None) -> list[Chunk]:
    return get_index().vector_search(query, k or settings.top_k_per_source)


@traceable(name="retrieve:lexical", run_type="retriever")
def lexical_source(query: str, k: int | None = None) -> list[Chunk]:
    return get_index().lexical_search(query, k or settings.top_k_per_source)


@traceable(name="retrieve:web", run_type="retriever")
def web_source(query: str, k: int | None = None) -> list[Chunk]:
    """Claude's server-side web_search tool, called through the official SDK."""
    if not settings.enable_web:
        return []
    k = k or settings.top_k_per_source

    tool: dict = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
    if settings.web_allowed_domains:
        tool["allowed_domains"] = list(settings.web_allowed_domains)

    try:
        response = raw_client().messages.create(
            model=settings.model,
            max_tokens=4096,
            output_config={"effort": "low"},
            tools=[tool],
            messages=[{
                "role": "user",
                "content": (
                    "Search the web and report only what the sources say about this, "
                    "with no commentary of your own:\n\n" + query
                ),
            }],
        )
    except Exception as exc:  # noqa: BLE001 -- a dead source must not kill the graph
        return [Chunk(id="web#error", text="web search unavailable: " + str(exc),
                      source="web", origin="web", title="web", score=0.0)]

    chunks: list[Chunk] = []
    for block in response.content:
        if block.type != "web_search_tool_result":
            continue
        # Success => content is a list of results; an error => content is an object.
        results = block.content if isinstance(block.content, list) else []
        for i, result in enumerate(results[:k]):
            body = getattr(result, "encrypted_content", "") or ""
            chunks.append(
                Chunk(
                    id="web#" + str(len(chunks)),
                    text=(getattr(result, "title", "") + "\n" + body)[:2000],
                    source="web",
                    origin=getattr(result, "url", "web"),
                    title=getattr(result, "title", "web result"),
                    score=1.0 - i * 0.05,
                )
            )
    # Claude's own synthesis of the results is itself a usable chunk.
    summary = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if summary:
        chunks.append(Chunk(id="web#" + str(len(chunks)), text=summary, source="web",
                            origin="web-search-summary", title="web synthesis", score=0.5))
    return chunks[: k + 1]


SOURCES = {"vector": vector_source, "lexical": lexical_source, "web": web_source}


@traceable(name="retrieve:fan-out")
def retrieve(query: str, sources: list[str], k: int | None = None) -> list[Chunk]:
    """Query every requested source and fuse the results (reciprocal rank fusion)."""
    ranked: dict[str, tuple[Chunk, float]] = {}
    for name in sources:
        fn = SOURCES.get(name)
        if fn is None:
            continue
        for rank, chunk in enumerate(fn(query, k)):
            rrf = 1.0 / (60 + rank)
            if chunk.id in ranked:
                prev, score = ranked[chunk.id]
                ranked[chunk.id] = (prev, score + rrf)
            else:
                ranked[chunk.id] = (chunk, rrf)
    fused = sorted(ranked.values(), key=lambda pair: -pair[1])
    limit = (k or settings.top_k_per_source) * max(1, len(sources))
    return [replace(chunk, score=score) for chunk, score in fused[:limit]]
