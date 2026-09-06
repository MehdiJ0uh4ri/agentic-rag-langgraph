"""Corpus loading and chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.config import settings

CHUNK_TARGET = 1100
CHUNK_OVERLAP_PARAGRAPHS = 1


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str          # which retriever produced it: vector | lexical | web
    origin: str          # file path, url, ...
    title: str
    score: float = 0.0

    def render(self) -> str:
        return f"[{self.id}] ({self.origin})\n{self.text}"


def _split(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) > CHUNK_TARGET and buf:
            chunks.append("\n\n".join(buf))
            buf = buf[-CHUNK_OVERLAP_PARAGRAPHS:] if CHUNK_OVERLAP_PARAGRAPHS else []
            size = sum(len(p) for p in buf)
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def load_corpus(root: Path | None = None) -> list[Chunk]:
    """Read every markdown/text file under `root` and chunk it."""
    root = root or settings.corpus_dir
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    chunks: list[Chunk] = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        title = raw.lstrip().splitlines()[0].lstrip("# ").strip() if raw.strip() else path.stem
        rel = path.relative_to(root).as_posix()
        for i, body in enumerate(_split(raw)):
            chunks.append(
                Chunk(
                    id=f"{rel}#{i}",
                    text=body,
                    source="corpus",
                    origin=rel,
                    title=title,
                )
            )
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]
