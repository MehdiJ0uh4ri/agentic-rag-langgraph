.PHONY: install index ask test eval eval-cheap lint

install:
	python -m pip install -e ".[dev]"

index:
	python -m agentic_rag.cli index

ask:
	python -m agentic_rag.cli ask "$(Q)"

test:
	python -m pytest -q

eval:
	python evals/run_evals.py --experiment $(or $(EXP),baseline)

eval-cheap:
	python evals/run_evals.py --no-judges --experiment $(or $(EXP),deterministic)

lint:
	python -m ruff check src evals tests
