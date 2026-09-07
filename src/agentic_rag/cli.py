"""Command line entry point: `arag ask "..."`."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agentic_rag.config import settings
from agentic_rag.graph import run
from agentic_rag.retrieval import get_index

app = typer.Typer(add_completion=False, help="Agentic RAG over LangGraph.")
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    show_plan: bool = typer.Option(True, help="Print the query decomposition."),
    show_grades: bool = typer.Option(True, help="Print per-sub-question context grades."),
    as_json: bool = typer.Option(False, "--json", help="Emit the final state as JSON."),
) -> None:
    if not settings.tracing_enabled:
        console.print("[yellow]LangSmith tracing is off[/] (set LANGSMITH_TRACING=true "
                      "and LANGSMITH_API_KEY to trace every call).")

    state = run(question)

    if as_json:
        payload = {
            "question": question,
            "coverage": state.get("coverage"),
            "decision": state.get("decision"),
            "answer": state["answer"].model_dump() if state.get("answer") else None,
            "refusal": state["refusal"].model_dump() if state.get("refusal") else None,
        }
        console.print_json(json.dumps(payload))
        raise typer.Exit()

    if show_plan and state.get("plan"):
        plan = state["plan"]
        table = Table(title="Query plan: " + plan.intent, show_lines=False)
        table.add_column("id"); table.add_column("sub-question"); table.add_column("sources")
        for sq in plan.sub_questions:
            table.add_row(sq.id, sq.text, ",".join(sq.sources))
        console.print(table)

    if show_grades and state.get("grades"):
        table = Table(title="Context grades (coverage " + format(state.get("coverage", 0), ".0%") + ")")
        table.add_column("id"); table.add_column("suff."); table.add_column("relevant")
        table.add_column("missing")
        for g in state["grades"]:
            table.add_row(g.sub_question_id, format(g.sufficiency, ".2f"),
                          str(sum(1 for c in g.grades if c.relevant)) + "/" + str(len(g.grades)),
                          g.missing or "-")
        console.print(table)

    if state.get("refusal"):
        r = state["refusal"]
        body = r.reason
        if r.partial_findings:
            body += "\n\n**What the corpus does support:** " + r.partial_findings
        if r.missing_evidence:
            body += "\n\n**Missing:**\n" + "\n".join("- " + m for m in r.missing_evidence)
        if r.suggested_rephrasing:
            body += "\n\n*Try instead:* " + r.suggested_rephrasing
        console.print(Panel(Markdown(body), title="Refused", border_style="red"))
    elif state.get("answer"):
        a = state["answer"]
        console.print(Panel(Markdown(a.answer), title="Answer", border_style="green"))
        cites = Table(title="Citations")
        cites.add_column("chunk"); cites.add_column("quote")
        for c in a.citations:
            cites.add_row(c.chunk_id, c.quote[:160])
        console.print(cites)
        if a.caveats:
            console.print(Panel(a.caveats, title="Caveats", border_style="yellow"))

    console.print("[dim]" + " | ".join(state.get("notes", [])) + "[/dim]")


@app.command()
def index() -> None:
    """Show what the retriever loaded."""
    idx = get_index()
    console.print("backend: [bold]" + idx.backend + "[/]")
    console.print("chunks: [bold]" + str(len(idx.chunks)) + "[/] from " + str(settings.corpus_dir))
    for c in idx.chunks[:20]:
        console.print("  " + c.id + "  " + c.text[:70].replace("\n", " ") + "...")


if __name__ == "__main__":
    app()
