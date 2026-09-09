# Agentic RAG with LangGraph

A retrieval agent that **decomposes** a query into sub-questions, **retrieves** each one
from multiple sources, **grades its own context**, and **refuses** when the evidence is
not there — instead of producing a fluent guess. Every LLM and retriever call is traced
to LangSmith, and the whole thing is scored by an eval suite that treats *"refused when it
should have"* as a first-class metric.

Model: `claude-opus-5` throughout, at three effort levels (planning and answering at
`high`, grading at `low`). Adaptive thinking is on everywhere.

---

## The graph

```
plan ─┬─(unanswerable in principle)──────────────────────► refuse ► END
      └► retrieve ► grade ─┬─(coverage ≥ threshold)─► answer ► verify ─┬─(grounded)─► END
                           │                                           └─(hallucinated)─┐
                           ├─(gaps, rounds left)─► bump ─┐                               │
                           │                             └──► retrieve                   │
                           └─(gaps, no rounds left)──────────────────────────► refuse ◄──┘
```

There are **three separate refusal paths**, because there are three distinct ways to not
know something:

| Path | Trigger | Node |
|---|---|---|
| Unaskable | The planner marks the query as one no corpus could settle (opinion, prediction, personal data) | `plan → refuse`, before any retrieval |
| Unretrievable | Context grades stay below the sufficiency floor after every retrieval round | `grade → refuse` |
| Ungroundable | An answer was drafted, but the groundedness check found claims beyond the citations | `verify → refuse` |

### Nodes

- **`plan`** — [`QueryPlan`](src/agentic_rag/schemas.py) structured output: intent, an
  `is_answerable_in_principle` flag, and up to four self-contained sub-questions. The
  planner also picks *which sources* each sub-question should hit.
- **`retrieve`** — fans out over the sub-questions and, for each, over its chosen sources.
  Results are fused with reciprocal rank fusion. On a retry round it only re-searches the
  sub-questions that failed, using the grader's rewritten query.
- **`grade`** — one [`ContextGrade`](src/agentic_rag/schemas.py) per sub-question, run
  concurrently via `.batch()`. Per-chunk relevance verdicts, a `sufficiency` score, the
  named `missing` evidence, and a `followup_query` for the next round. **A grader call
  that throws is scored 0.0, never a pass** — the failure mode of a self-grading system is
  a grader that silently succeeds.
- **`answer`** — [`Answer`](src/agentic_rag/schemas.py) with citations whose `quote` must
  be a verbatim span, which is what makes citation validity checkable offline.
- **`verify`** — an independent groundedness pass over the draft and its citations.
- **`refuse`** — [`Refusal`](src/agentic_rag/schemas.py): what's missing, and any
  `partial_findings` that *were* genuinely established. A partial answer beats a bare no.

Every schema is bound with `.with_structured_output(...)`, so no node parses prose.

### Sources

| Source | Backing | Best for |
|---|---|---|
| `vector` | `fastembed` / BGE-small, cosine over the chunked corpus | conceptual overlap |
| `lexical` | BM25 (`rank_bm25`) | exact identifiers, error codes, proper nouns |
| `web` | Claude's server-side `web_search_20260209` tool, via the official `anthropic` SDK | facts outside the corpus |

If `fastembed` can't load, the index falls back to a deterministic hashing TF-IDF
embedder — same interface, lower quality, keeps the pipeline runnable offline. A failing
web search returns an error chunk rather than raising: **a dead source must not kill the
graph.**

---

## Setup

```bash
make install                  # pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY + LANGSMITH_API_KEY
```

Credentials also resolve from an `ant auth login` profile if no key is exported.

```bash
make index                    # what the retriever loaded
make ask Q="I want to expense a EUR 900 conference ticket. Who approves it?"
make test                     # offline, no API key needed
```

`arag ask` prints the query plan, the per-sub-question grades with their sufficiency
scores, and then the answer with citations — or the refusal with its missing-evidence
list. `--json` emits the final state instead.

---

## LangSmith tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; the CLI warns when tracing is off.
Each run is one trace named `agentic-rag`, carrying the model, coverage threshold and
round budget as metadata. Inside it:

- one span per LLM node, tagged `planner` / `grader` / `answerer` / `groundedness` /
  `refuser`, so you can filter cost by role;
- one `run_type="retriever"` span per source per sub-question (`retrieve:vector`,
  `retrieve:lexical`, `retrieve:web`) nested under `retrieve:fan-out` — non-LLM steps are
  traced too, which is where retrieval regressions actually show up.

---

## Evals

The dataset ([`evals/dataset.py`](evals/dataset.py)) has three families, because an
agentic RAG system fails in three ways:

- **answerable** — the corpus holds the answer, including one multi-hop question spanning
  two documents. Failing = refusing, or dropping a numeric threshold.
- **unanswerable** — deliberately *adjacent* questions. The corpus discusses per diems but
  states international rates live elsewhere; it gives recording retention for broker
  sessions but not break-glass accounts. This is where a naive RAG pipeline confidently
  answers from the nearest-looking chunk.
- **unaskable** — an opinion question. No retrieval should be attempted at all.

Evaluators ([`evals/evaluators.py`](evals/evaluators.py)) — deterministic first, since
those are the ones to trust:

| Metric | Kind | What it catches |
|---|---|---|
| `refusal_decision` | deterministic | The headline metric: answered vs. refused vs. what was expected |
| `citation_validity` | deterministic | Fabricated chunk ids and non-verbatim quotes, checked against the corpus |
| `context_recall` | deterministic | Whether retrieval surfaced the gold documents at all |
| `decomposition_shape` | deterministic | Empty, duplicate, or dangling-pronoun sub-questions |
| `answer_correctness` | LLM judge | Semantic match to a reference answer; missing thresholds penalised |
| `refusal_quality` | LLM judge | Vague refusals, and refusals that smuggle in a guess |

```bash
make eval                     # full suite
make eval-cheap               # deterministic metrics only, no judge calls
make eval EXP=stricter-grader # named experiment for A/B comparison in LangSmith
```

Experiment metadata records `grader_effort`, `min_coverage`, `min_sufficiency` and
`max_retrieval_rounds`, which are the knobs worth sweeping — they trade the answerable
family against the unanswerable one. Raising `min_sufficiency` buys refusal precision at
the cost of recall on answerable questions; the eval suite is what tells you where that
line should sit.

---

## Tuning

Everything is env-driven ([`src/agentic_rag/config.py`](src/agentic_rag/config.py)):

| Variable | Default | Effect |
|---|---|---|
| `ARAG_MIN_SUFFICIENCY` | `0.6` | Per-sub-question bar for "answerable from this context" |
| `ARAG_MIN_COVERAGE` | `0.75` | Share of sub-questions that must clear that bar |
| `ARAG_MAX_RETRIEVAL_ROUNDS` | `2` | Retrieve→grade→rewrite attempts before refusing |
| `ARAG_GRADER_EFFORT` | `low` | Effort for grader/judge calls |
| `ARAG_ENABLE_WEB` | `true` | Turn off the web source for a corpus-only run |

---

## Tests

15 offline tests ([`tests/test_graph.py`](tests/test_graph.py)) with every LLM node
stubbed. They pin the control flow — that a retry exhausts its budget before refusing,
that an ungrounded draft is discarded rather than shipped, that the agent never drafts an
answer it can't ground — plus the retrieval fusion, the state reducer, and the
deterministic evaluators.

```bash
make test    # 15 passed, no API key required
make lint
```

## Notes

- Thinking is left adaptive on every call. Disabling it on Opus 5 lets tool calls leak
  into visible text, which silently breaks structured output; lowering `effort` is the
  correct cost lever, and that's what the graders use.
- `retrieved` uses a merge reducer, so a second retrieval round *adds* to what the first
  found rather than replacing it — the grader sees the union.
