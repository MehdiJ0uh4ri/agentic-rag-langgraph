"""The eval dataset, pushed to LangSmith on demand.

Three families of example, because an agentic RAG system can fail in three ways:

* `answerable`   -- the corpus contains the answer. Failing = refusing or getting it wrong.
* `unanswerable` -- the corpus does NOT contain it. Failing = answering anyway.
* `unaskable`    -- no corpus could answer it. Failing = pretending to retrieve.
"""

from __future__ import annotations

from langsmith import Client

DATASET_NAME = "agentic-rag-eval"

EXAMPLES: list[dict] = [
    # ---- answerable ------------------------------------------------------ #
    {
        "question": "I want to expense a EUR 900 conference ticket. Who has to approve it, and how long do they have?",
        "expected": "answer",
        "reference": ("Between EUR 250 and EUR 2,000, so line-manager approval is required. "
                      "Approvals must be granted within five working days, and unapproved "
                      "claims older than 30 days close automatically."),
        "expected_docs": ["handbook/expenses.md"],
    },
    {
        "question": "Can a contractor who joined last week get production database access with a TOTP app?",
        "expected": "answer",
        "reference": ("TOTP is accepted for contractors only during their first 30 days, so yes "
                      "initially, but a FIDO2 hardware key must be enrolled after that. Production "
                      "database access is just-in-time via the access broker, max four hours."),
        "expected_docs": ["security/access-control.md"],
    },
    {
        "question": "It's Thursday afternoon and error rate just hit 3%. What happens to my deploy, and how fast can I roll back manually?",
        "expected": "answer",
        "reference": ("Above 2% error rate for five consecutive minutes the progressive delivery "
                      "controller rolls back automatically. Manual rollback via `deployctl rollback "
                      "--release <id>` takes effect within 90 seconds."),
        "expected_docs": ["platform/deployments.md"],
    },
    {
        "question": "How many annual leave days can I carry over, and by when must I use them?",
        "expected": "answer",
        "reference": "Up to 5 unused days carry over and must be used before 31 March, after which they lapse.",
        "expected_docs": ["handbook/leave.md"],
    },
    # Multi-hop: needs two documents.
    {
        "question": "I'm taking three weeks off starting the Monday of a release freeze. What notice and approvals do I need?",
        "expected": "answer",
        "reference": ("21 days' notice for absences longer than five consecutive days, plus director "
                      "approval because it falls in the two weeks around a release freeze."),
        "expected_docs": ["handbook/leave.md", "platform/deployments.md"],
    },
    # ---- unanswerable: plausible, adjacent, but absent -------------------- #
    {
        "question": "What is the per diem rate for travel to Japan?",
        "expected": "refuse",
        "reference": ("The corpus states international per diem rates live in the Finance wiki and "
                      "are not reproduced. Only domestic rates (EUR 45 / EUR 22) are given."),
        "expected_docs": ["handbook/expenses.md"],
    },
    {
        "question": "How long are break-glass account session recordings kept, and who reviews them?",
        "expected": "refuse",
        "reference": ("Recording retention (400 days) is stated for privileged broker sessions, not "
                      "for break-glass accounts, and no reviewer is named."),
        "expected_docs": ["security/access-control.md"],
    },
    {
        "question": "What is the maximum number of canary stages allowed for a database migration?",
        "expected": "refuse",
        "reference": "The runbook describes a fixed 5/25/100 canary for releases and says nothing about migrations.",
        "expected_docs": ["platform/deployments.md"],
    },
    # ---- unaskable ------------------------------------------------------- #
    {
        "question": "Should we switch the whole company to a four-day week next year?",
        "expected": "refuse",
        "reference": "An opinion/decision request; no document can settle it.",
        "expected_docs": [],
    },
]


def to_langsmith(client: Client | None = None) -> str:
    """Create or update the LangSmith dataset. Returns the dataset name."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Answerable / unanswerable / unaskable questions for the agentic RAG graph.",
        )

    existing = {ex.inputs.get("question") for ex in client.list_examples(dataset_id=dataset.id)}
    new = [e for e in EXAMPLES if e["question"] not in existing]
    if new:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": e["question"]} for e in new],
            outputs=[
                {
                    "expected": e["expected"],
                    "reference": e["reference"],
                    "expected_docs": e["expected_docs"],
                }
                for e in new
            ],
        )
    return DATASET_NAME
